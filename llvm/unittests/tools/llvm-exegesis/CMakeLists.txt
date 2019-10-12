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
  BenchmarkRunnerTest.cpp
  ClusteringTest.cpp
  PerfHelperTest.cpp
  RegisterValueTest.cpp
  )
target_link_libraries(LLVMExegesisTests PRIVATE LLVMExegesis)

if(LLVM_TARGETS_TO_BUILD MATCHES "X86")
  add_subdirectory(X86)
endif()
if(LLVM_TARGETS_TO_BUILD MATCHES "ARM")
	add_subdirectory(ARM)
endif()
if(LLVM_TARGETS_TO_BUILD MATCHES "AArch64")
  add_subdirectory(AArch64)
endif()
if(LLVM_TARGETS_TO_BUILD MATCHES "PowerPC")
  add_subdirectory(PowerPC)
endif()
if(LLVM_TARGETS_TO_BUILD MATCHES "Mips")
  add_subdirectory(Mips)
endif()
