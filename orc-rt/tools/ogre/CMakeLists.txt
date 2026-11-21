add_executable(ogre ogre.cpp)
target_compile_options(ogre PRIVATE ${ORC_RT_COMPILE_FLAGS})

# Export ogre's symbols to its dynamic symbol table (-rdynamic on ELF), so that
# JIT'd code can resolve the runtime's C API symbols (the ORC_RT_C_EXPORT
# declarations under include/orc-rt-c/) against the ogre process.
set_target_properties(ogre PROPERTIES ENABLE_EXPORTS ON)

# ogre uses the C++ API, which the shipped libraries do not export when built
# as shared libraries, so link the objects rather than the shipped libraries.
target_link_libraries(ogre
  PRIVATE orc-rt-connect-tcp-objects orc-rt-connect-socket-objects
          orc-rt-connect-objects orc-rt-bedrock-objects orc-rt-support-objects)
