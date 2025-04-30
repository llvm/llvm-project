add_llvm_component_group(Next32)

set(LLVM_TARGET_DEFINITIONS Next32.td)

tablegen(LLVM Next32GenRegisterInfo.inc -gen-register-info)
tablegen(LLVM Next32GenInstrInfo.inc -gen-instr-info)
tablegen(LLVM Next32GenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM Next32GenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM Next32GenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM Next32GenDAGISel.inc -gen-dag-isel)
tablegen(LLVM Next32GenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM Next32GenCallingConv.inc -gen-callingconv)
tablegen(LLVM Next32GenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(Next32CommonTableGen)

add_llvm_target(Next32CodeGen
  Next32AsmPrinter.cpp
  Next32FrameLowering.cpp
  Next32InstrInfo.cpp
  Next32ISelDAGToDAG.cpp
  Next32ISelLowering.cpp
  Next32MCInstLower.cpp
  Next32Mem2Reg.cpp
  Next32RegisterInfo.cpp
  Next32Subtarget.cpp
  Next32CondBranchFixup.cpp
  Next32WriterChain.cpp
  Next32OrderCallChain.cpp
  Next32EliminateCallTerminators.cpp
  Next32CalculateFeeders.cpp
  Next32CallSplit.cpp
  Next32AddRetFid.cpp
  Next32CallTerminatorPass.cpp
  Next32TargetMachine.cpp
  Next32TargetTransformInfo.cpp
  Next32MachineFunctionInfo.cpp
  Next32TargetObjectFile.cpp
  Next32SelectionDAGInfo.cpp

  LINK_COMPONENTS
  AsmPrinter
  CodeGen
  Core
  MC
  Next32Desc
  Next32Info
  IPO
  SelectionDAG
  Support
  Target

  ADD_TO_COMPONENT
  Next32
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
