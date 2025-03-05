set(LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  Core
  Passes
  Support
  ScalarOpts
  TransformUtils
  )

add_llvm_unittest(ScalarTests
  LICMTest.cpp
  LoopPassManagerTest.cpp
  )

target_link_libraries(ScalarTests PRIVATE LLVMTestingSupport)

# Workaround for the gcc 6.1 bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80916.
if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6.0 AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
  set_source_files_properties(LoopPassManagerTest.cpp PROPERTIES COMPILE_FLAGS -Wno-unused-function)
endif()
