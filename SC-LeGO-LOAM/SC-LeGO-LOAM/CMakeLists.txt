cmake_minimum_required(VERSION 3.5)
project(sc_lego_loam)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "-std=c++11")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")

find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  rospy
  cv_bridge
  image_transport

  pcl_ros
  pcl_conversions

  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  cloud_msgs
)


set(GTSAM_INCLUDE_DIR /home/zhaochengwei/workspace/3rdparty/gtsam400/include)
set(GTSAM_LIBRARY_DIRS /home/zhaochengwei/workspace/3rdparty/gtsam400/lib)
file(GLOB GTSAM_LIBRARIES ${GTSAM_LIBRARY_DIRS}/*.a ${GTSAM_LIBRARY_DIRS}/*.so*)
message(STATUS "gtsam lib: ${GTSAM_LIBRARIES}")

find_package(PCL REQUIRED QUIET)
find_package(OpenCV REQUIRED QUIET)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS cloud_msgs
  DEPENDS PCL
)

include_directories(
	include
	${catkin_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${GTSAM_INCLUDE_DIR}
)

link_directories(
	include
	${OpenCV_LIBRARY_DIRS}
	${PCL_LIBRARY_DIRS}
	${GTSAM_LIBRARY_DIRS}
)

add_executable(imageProjection src/imageProjection.cpp)
add_dependencies(imageProjection ${catkin_EXPORTED_TARGETS} )
target_link_libraries(imageProjection ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})


add_executable(featureAssociation src/featureAssociation.cpp)
add_dependencies(featureAssociation ${catkin_EXPORTED_TARGETS} )
target_link_libraries(featureAssociation ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(mapOptmization 
    src/mapOptmization.cpp
    src/Scancontext.cpp
    src/Frame.cpp       
    src/LinK3D_Extractor.cpp 
    src/BoW3D.cpp 
)
target_link_libraries(mapOptmization 
    ${catkin_LIBRARIES} 
    ${PCL_LIBRARIES} 
    ${OpenCV_LIBRARIES} 
    gtsam
    libtbb.so
    # Eigen3::Eigen
)

add_executable(transformFusion src/transformFusion.cpp)
target_link_libraries(transformFusion ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})
