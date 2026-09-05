add_executable(ogre ogre.cpp)
target_compile_options(ogre PRIVATE ${ORC_RT_COMPILE_FLAGS})
target_link_libraries(ogre PRIVATE orc-rt-bedrock)
