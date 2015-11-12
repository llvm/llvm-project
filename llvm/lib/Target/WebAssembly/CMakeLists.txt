set(LLVM_TARGET_DEFINITIONS WebAssembly.td)

tablegen(LLVM WebAssemblyGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM WebAssemblyGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM WebAssemblyGenFastISel.inc -gen-fast-isel)
tablegen(LLVM WebAssemblyGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM WebAssemblyGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM WebAssemblyGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM WebAssemblyGenSubtargetInfo.inc -gen-subtarget)
add_public_tablegen_target(WebAssemblyCommonTableGen)

add_llvm_target(WebAssemblyCodeGen
  Relooper.cpp
  WebAssemblyAsmPrinter.cpp
  WebAssemblyCFGStackify.cpp
  WebAssemblyFastISel.cpp
  WebAssemblyFrameLowering.cpp
  WebAssemblyISelDAGToDAG.cpp
  WebAssemblyISelLowering.cpp
  WebAssemblyInstrInfo.cpp
  WebAssemblyMachineFunctionInfo.cpp
  WebAssemblyMCInstLower.cpp
  WebAssemblyRegisterInfo.cpp
  WebAssemblyRegNumbering.cpp
  WebAssemblySelectionDAGInfo.cpp
  WebAssemblySubtarget.cpp
  WebAssemblyTargetMachine.cpp
  WebAssemblyTargetTransformInfo.cpp
)

add_dependencies(LLVMWebAssemblyCodeGen intrinsics_gen)

add_subdirectory(InstPrinter)
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
