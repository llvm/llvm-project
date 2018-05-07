include_directories(
  ${LLVM_MAIN_SRC_DIR}/tools/llvm-exegesis/lib
  )

set(LLVM_LINK_COMPONENTS
  MC
  MCParser
  Object
  Support
  Symbolize
  )

add_llvm_unittest(LLVMExegesisTests
  BenchmarkResultTest.cpp
  ClusteringTest.cpp
  OperandGraphTest.cpp
  PerfHelperTest.cpp
  )
target_link_libraries(LLVMExegesisTests PRIVATE LLVMExegesis)

if(LLVM_ENABLE_LIBPFM AND HAVE_LIBPFM)
  target_link_libraries(LLVMExegesisTests PRIVATE pfm)
endif()

if(LLVM_TARGETS_TO_BUILD MATCHES "X86")
  add_subdirectory(
    X86
  )
endif()
