add_llvm_component_group(WebAssembly)

set(LLVM_TARGET_DEFINITIONS WebAssembly.td)

llvm_tablegen(WebAssemblyGenAsmMatcher.inc -gen-asm-matcher)
llvm_tablegen(WebAssemblyGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(WebAssemblyGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(WebAssemblyGenDisassemblerTables.inc -gen-disassembler)
llvm_tablegen(WebAssemblyGenFastISel.inc -gen-fast-isel)
llvm_tablegen(WebAssemblyGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(WebAssemblyGenMCCodeEmitter.inc -gen-emitter)
llvm_tablegen(WebAssemblyGenRegisterBank.inc -gen-register-bank)
llvm_tablegen(WebAssemblyGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(WebAssemblyGenSDNodeInfo.inc -gen-sd-node-info)
llvm_tablegen(WebAssemblyGenSubtargetInfo.inc -gen-subtarget)

set(LLVM_TARGET_DEFINITIONS WebAssemblyGISel.td)
llvm_tablegen(WebAssemblyGenGlobalISel.inc -gen-global-isel -gisel-extended-llt)
llvm_tablegen(WebAssemblyGenPreLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="WebAssemblyPreLegalizerCombiner")
llvm_tablegen(WebAssemblyGenPostLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="WebAssemblyPostLegalizerCombiner")

add_public_tablegen_target(WebAssemblyCommonTableGen)

add_llvm_target(WebAssemblyCodeGen
  GISel/WebAssemblyCallLowering.cpp
  GISel/WebAssemblyInstructionSelector.cpp
  GISel/WebAssemblyPostLegalizerCombiner.cpp
  GISel/WebAssemblyPreLegalizerCombiner.cpp
  GISel/WebAssemblyLegalizerInfo.cpp
  GISel/WebAssemblyRegisterBankInfo.cpp
  WebAssemblyAddMissingPrototypes.cpp
  WebAssemblyArgumentMove.cpp
  WebAssemblyAsmPrinter.cpp
  WebAssemblyCFGStackify.cpp
  WebAssemblyCleanCodeAfterTrap.cpp
  WebAssemblyCFGSort.cpp
  WebAssemblyCoalesceFeaturesAndStripAtomics.cpp
  WebAssemblyCodeGenPassBuilder.cpp
  WebAssemblyDebugFixup.cpp
  WebAssemblyDebugValueManager.cpp
  WebAssemblyLateEHPrepare.cpp
  WebAssemblyExceptionInfo.cpp
  WebAssemblyExplicitLocals.cpp
  WebAssemblyFastISel.cpp
  WebAssemblyFixBrTableDefaults.cpp
  WebAssemblyFixIrreducibleControlFlow.cpp
  WebAssemblyFixFunctionBitcasts.cpp
  WebAssemblyFrameLowering.cpp
  WebAssemblyISelDAGToDAG.cpp
  WebAssemblyISelLowering.cpp
  WebAssemblyInstrInfo.cpp
  WebAssemblyLowerBrUnless.cpp
  WebAssemblyLowerEmscriptenEHSjLj.cpp
  WebAssemblyMachineFunctionInfo.cpp
  WebAssemblyMCInstLower.cpp
  WebAssemblyMCLowerPrePass.cpp
  WebAssemblyNullifyDebugValueLists.cpp
  WebAssemblyOptimizeLiveIntervals.cpp
  WebAssemblyOptimizeReturned.cpp
  WebAssemblyPeephole.cpp
  WebAssemblyReduceToAnyAllTrue.cpp
  WebAssemblyRefTypeMem2Local.cpp
  WebAssemblyRegisterInfo.cpp
  WebAssemblyRegColoring.cpp
  WebAssemblyRegNumbering.cpp
  WebAssemblyRegStackify.cpp
  WebAssemblyReplacePhysRegs.cpp
  WebAssemblyRuntimeLibcallSignatures.cpp
  WebAssemblySelectionDAGInfo.cpp
  WebAssemblySetP2AlignOperands.cpp
  WebAssemblySortRegion.cpp
  WebAssemblyMemIntrinsicResults.cpp
  WebAssemblySubtarget.cpp
  WebAssemblyTargetMachine.cpp
  WebAssemblyTargetObjectFile.cpp
  WebAssemblyTargetTransformInfo.cpp
  WebAssemblyUtilities.cpp

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  BinaryFormat
  CodeGen
  CodeGenTypes
  Core
  GlobalISel
  MC
  Passes
  Scalar
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils
  WebAssemblyDesc
  WebAssemblyInfo
  WebAssemblyUtils

  ADD_TO_COMPONENT
  WebAssembly
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
add_subdirectory(Utils)
