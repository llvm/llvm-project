add_llvm_component_group(DirectX)

set(LLVM_TARGET_DEFINITIONS DirectX.td)

tablegen(LLVM DirectXGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM DirectXGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM DirectXGenRegisterInfo.inc -gen-register-info)

set(LLVM_TARGET_DEFINITIONS DXIL.td)
tablegen(LLVM DXILOperation.inc -gen-dxil-operation)

add_public_tablegen_target(DirectXCommonTableGen)

add_llvm_target(DirectXCodeGen
  CBufferDataLayout.cpp
  DirectXAsmPrinter.cpp
  DirectXInstrInfo.cpp
  DirectXRegisterInfo.cpp
  DirectXSubtarget.cpp
  DirectXTargetMachine.cpp
  DXContainerGlobals.cpp
  DXILMetadata.cpp
  DXILOpBuilder.cpp
  DXILOpLowering.cpp
  DXILPrepare.cpp
  DXILPrettyPrinter.cpp
  DXILResource.cpp
  DXILResourceAnalysis.cpp
  DXILShaderFlags.cpp
  DXILTranslateMetadata.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  Core
  CodeGen
  SelectionDAG
  Support
  TargetParser
  DirectXDesc
  DirectXInfo
  DirectXPointerTypeAnalysis
  DXILBitWriter
  FrontendHLSL
  MC
  Target
  TransformUtils

  ADD_TO_COMPONENT
  DirectX
)

add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
add_subdirectory(DirectXIRPasses)
add_subdirectory(DXILWriter)
