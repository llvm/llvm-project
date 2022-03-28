add_llvm_component_group(DirectX)

set(LLVM_TARGET_DEFINITIONS DirectX.td)

tablegen(LLVM DirectXGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(DirectXCommonTableGen)

add_llvm_target(DirectXCodeGen
  DirectXSubtarget.cpp
  DirectXTargetMachine.cpp

  LINK_COMPONENTS
  Bitwriter
  Core
  Support
  DirectXInfo

  ADD_TO_COMPONENT
  DirectX
)

add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
