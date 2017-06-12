set(LLVM_LINK_COMPONENTS
  DebugInfoCodeView
  DebugInfoPDB
  Object
  Support
  )

add_llvm_executable(llvm-pdbutil-fuzzer
  EXCLUDE_FROM_ALL
  llvm-pdbutil-fuzzer.cpp
  )

target_link_libraries(llvm-pdbutil-fuzzer
  LLVMFuzzer
  )
