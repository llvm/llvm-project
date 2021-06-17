include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/AMDGPU
  ${LLVM_BINARY_DIR}/lib/Target/AMDGPU
  )

set(LLVM_LINK_COMPONENTS
  AMDGPU
  Core
  MCA
  Support
  )

add_llvm_library(LLVMMCACustomBehaviourAMDGPU
  AMDGPUCustomBehaviour.cpp

  DEPENDS
  AMDGPUCommonTableGen
  )
