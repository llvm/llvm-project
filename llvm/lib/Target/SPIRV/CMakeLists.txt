add_llvm_component_group(SPIRV)

set(LLVM_TARGET_DEFINITIONS SPIRV.td)

tablegen(LLVM SPIRVGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM SPIRVGenGlobalISel.inc -gen-global-isel)
tablegen(LLVM SPIRVGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM SPIRVGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM SPIRVGenRegisterBank.inc -gen-register-bank)
tablegen(LLVM SPIRVGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM SPIRVGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM SPIRVGenTables.inc -gen-searchable-tables)
tablegen(LLVM SPIRVGenPreLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="SPIRVPreLegalizerCombiner")

add_public_tablegen_target(SPIRVCommonTableGen)

add_llvm_target(SPIRVCodeGen
  SPIRVAPI.cpp
  SPIRVAsmPrinter.cpp
  SPIRVBuiltins.cpp
  SPIRVCallLowering.cpp
  SPIRVInlineAsmLowering.cpp
  SPIRVCommandLine.cpp
  SPIRVEmitIntrinsics.cpp
  SPIRVGlobalRegistry.cpp
  SPIRVInstrInfo.cpp
  SPIRVInstructionSelector.cpp
  SPIRVLegalizeImplicitBinding.cpp
  SPIRVStripConvergentIntrinsics.cpp
  SPIRVLegalizePointerCast.cpp
  SPIRVMergeRegionExitTargets.cpp
  SPIRVISelLowering.cpp
  SPIRVLegalizerInfo.cpp
  SPIRVMCInstLower.cpp
  SPIRVMetadata.cpp
  SPIRVModuleAnalysis.cpp
  SPIRVStructurizer.cpp
  SPIRVCombinerHelper.cpp
  SPIRVPreLegalizer.cpp
  SPIRVPreLegalizerCombiner.cpp
  SPIRVPostLegalizer.cpp
  SPIRVPrepareFunctions.cpp
  SPIRVRegisterBankInfo.cpp
  SPIRVRegisterInfo.cpp
  SPIRVRegularizer.cpp
  SPIRVSubtarget.cpp
  SPIRVTargetMachine.cpp
  SPIRVTargetTransformInfo.cpp
  SPIRVUtils.cpp
  SPIRVEmitNonSemanticDI.cpp
  SPIRVCBufferAccess.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  Demangle
  GlobalISel
  FrontendHLSL
  MC
  SPIRVAnalysis
  SPIRVDesc
  SPIRVInfo
  ScalarOpts
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT
  SPIRV
  )

add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
add_subdirectory(Analysis)
