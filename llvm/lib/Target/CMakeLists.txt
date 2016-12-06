list(APPEND LLVM_COMMON_DEPENDS intrinsics_gen)

list(APPEND LLVM_TABLEGEN_FLAGS -I ${LLVM_MAIN_SRC_DIR}/lib/Target)

add_llvm_library(LLVMTarget
  Target.cpp
  TargetIntrinsicInfo.cpp
  TargetLoweringObjectFile.cpp
  TargetMachine.cpp
  TargetMachineC.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Target
  )

foreach(t ${LLVM_TARGETS_TO_BUILD})
  message(STATUS "Targeting ${t}")
  add_subdirectory(${t})
endforeach()
