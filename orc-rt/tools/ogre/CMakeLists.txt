add_executable(ogre ogre.cpp)
target_compile_options(ogre PRIVATE ${ORC_RT_COMPILE_FLAGS})
if (APPLE OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_compile_definitions(ogre PRIVATE ORC_RT_HAS_SOCKET_CONNECTOR)
endif()

# ogre uses the C++ API, which orc-rt-bedrock does not export when built as a
# shared library, so link the objects rather than the shipped library.
target_link_libraries(ogre
  PRIVATE orc-rt-bedrock-objects orc-rt-support-objects)
