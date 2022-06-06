add_llvm_component_group(DirectX)

set(LLVM_TARGET_DEFINITIONS DirectX.td)

tablegen(LLVM DirectXGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM DirectXGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM DirectXGenRegisterInfo.inc -gen-register-info)

set(LLVM_TARGET_DEFINITIONS DXIL.td)
tablegen(LLVM DXILOperation.inc -gen-dxil-operation)

add_public_tablegen_target(DirectXCommonTableGen)

add_llvm_target(DirectXCodeGen
  DirectXAsmPrinter.cpp
  DirectXInstrInfo.cpp
  DirectXRegisterInfo.cpp
  DirectXSubtarget.cpp
  DirectXTargetMachine.cpp
  DXILOpLowering.cpp
  DXILPointerType.cpp
  DXILPrepare.cpp
  DXILTranslateMetadata.cpp
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
