add_executable(ogre ogre.cpp)
target_compile_options(ogre PRIVATE ${ORC_RT_COMPILE_FLAGS})

# ogre uses the C++ API, which the shipped libraries do not export when built
# as shared libraries, so link the objects rather than the shipped libraries.
target_link_libraries(ogre
  PRIVATE orc-rt-connect-socket-objects orc-rt-connect-objects
          orc-rt-bedrock-objects orc-rt-support-objects)
