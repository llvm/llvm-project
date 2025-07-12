add_llvm_component_library(LLVMABI
  ABIFunctionInfo.cpp
  ABITypeMapper.cpp
  TargetCodeGenInfo.cpp
  Targets/BPF.cpp
  Targets/X86.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ABI

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  Support
)
