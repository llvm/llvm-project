include_directories(
  ${PROJECT_SOURCE_DIR}/lib/Target/WebAssembly
  ${PROJECT_BINARY_DIR}/lib/Target/WebAssembly
  )

set(LLVM_LINK_COMPONENTS
  CodeGen
  CodeGenTypes
  Core
  MC
  MIRParser
  TargetParser
  WebAssemblyCodeGen
  WebAssemblyDesc
  WebAssemblyInfo
  )

add_llvm_target_unittest(WebAssemblyTests
  WebAssemblyExceptionInfoTest.cpp
  )

set_property(TARGET WebAssemblyTests PROPERTY FOLDER "Tests/UnitTests/TargetTests")
