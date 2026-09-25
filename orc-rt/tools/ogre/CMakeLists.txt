add_executable(ogre ogre.cpp)
target_compile_options(ogre PRIVATE ${ORC_RT_COMPILE_FLAGS})
# ogre uses the C++ API, which orc-rt-bedrock does not export when built as a
# shared library, so link the objects rather than the shipped library.
target_link_libraries(ogre
  PRIVATE orc-rt-bedrock-objects orc-rt-support-objects)
