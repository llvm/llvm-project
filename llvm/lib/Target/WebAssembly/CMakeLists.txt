set(LLVM_TARGET_DEFINITIONS WebAssembly.td)

tablegen(LLVM WebAssemblyGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM WebAssemblyGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM WebAssemblyGenFastISel.inc -gen-fast-isel)
tablegen(LLVM WebAssemblyGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM WebAssemblyGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM WebAssemblyGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM WebAssemblyGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM WebAssemblyGenAsmMatcher.inc -gen-asm-matcher)
add_public_tablegen_target(WebAssemblyCommonTableGen)

add_llvm_target(WebAssemblyCodeGen
  WebAssemblyArgumentMove.cpp
  WebAssemblyAsmPrinter.cpp
  WebAssemblyCallIndirectFixup.cpp
  WebAssemblyCFGStackify.cpp
  WebAssemblyCFGSort.cpp
  WebAssemblyExplicitLocals.cpp
  WebAssemblyFastISel.cpp
  WebAssemblyFixIrreducibleControlFlow.cpp
  WebAssemblyFixFunctionBitcasts.cpp
  WebAssemblyFrameLowering.cpp
  WebAssemblyISelDAGToDAG.cpp
  WebAssemblyISelLowering.cpp
  WebAssemblyInstrInfo.cpp
  WebAssemblyLowerBrUnless.cpp
  WebAssemblyLowerEmscriptenEHSjLj.cpp
  WebAssemblyLowerGlobalDtors.cpp
  WebAssemblyMachineFunctionInfo.cpp
  WebAssemblyMCInstLower.cpp
  WebAssemblyOptimizeLiveIntervals.cpp
  WebAssemblyOptimizeReturned.cpp
  WebAssemblyPeephole.cpp
  WebAssemblyPrepareForLiveIntervals.cpp
  WebAssemblyRegisterInfo.cpp
  WebAssemblyRegColoring.cpp
  WebAssemblyRegNumbering.cpp
  WebAssemblyRegStackify.cpp
  WebAssemblyReplacePhysRegs.cpp
  WebAssemblyRuntimeLibcallSignatures.cpp
  WebAssemblySelectionDAGInfo.cpp
  WebAssemblySetP2AlignOperands.cpp
  WebAssemblyStoreResults.cpp
  WebAssemblySubtarget.cpp
  WebAssemblyTargetMachine.cpp
  WebAssemblyTargetObjectFile.cpp
  WebAssemblyTargetTransformInfo.cpp
  WebAssemblyUtilities.cpp

  DEPENDS
  intrinsics_gen
)

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(InstPrinter)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
