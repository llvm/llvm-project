set(LLVM_LINK_COMPONENTS
  DebugInfoCodeView
  DebugInfoPDB
  Object
  Support
  )

add_llvm_executable(llvm-pdbdump-fuzzer
  EXCLUDE_FROM_ALL
  llvm-pdbdump-fuzzer.cpp
  )

target_link_libraries(llvm-pdbdump-fuzzer
  LLVMFuzzer
  )
