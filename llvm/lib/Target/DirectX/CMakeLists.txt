add_llvm_component_group(DirectX)

set(LLVM_TARGET_DEFINITIONS DirectX.td)

tablegen(LLVM DirectXGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(DirectXCommonTableGen)

add_llvm_target(DirectXCodeGen
  DirectXSubtarget.cpp
  DirectXTargetMachine.cpp
  DXILOpLowering.cpp
  DXILPointerType.cpp
  DXILPrepare.cpp
  PointerTypeAnalysis.cpp

  LINK_COMPONENTS
  Core
  Support
  DirectXInfo
  DXILBitWriter

  ADD_TO_COMPONENT
  DirectX
)

add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
add_subdirectory(DXILWriter)
