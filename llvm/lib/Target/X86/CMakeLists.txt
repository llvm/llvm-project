set(LLVM_TARGET_DEFINITIONS X86.td)

tablegen(LLVM X86GenRegisterInfo.inc -gen-register-info)
tablegen(LLVM X86GenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM X86GenInstrInfo.inc -gen-instr-info)
tablegen(LLVM X86GenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM X86GenAsmWriter1.inc -gen-asm-writer -asmwriternum=1)
tablegen(LLVM X86GenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM X86GenDAGISel.inc -gen-dag-isel)
tablegen(LLVM X86GenFastISel.inc -gen-fast-isel)
tablegen(LLVM X86GenCallingConv.inc -gen-callingconv)
tablegen(LLVM X86GenSubtargetInfo.inc -gen-subtarget)
add_public_tablegen_target(X86CommonTableGen)

set(sources
  X86AsmPrinter.cpp
  X86CallFrameOptimization.cpp
  X86ExpandPseudo.cpp
  X86FastISel.cpp
  X86FloatingPoint.cpp
  X86FrameLowering.cpp
  X86ISelDAGToDAG.cpp
  X86ISelLowering.cpp
  X86InstrInfo.cpp
  X86MCInstLower.cpp
  X86MachineFunctionInfo.cpp
  X86PadShortFunction.cpp
  X86RegisterInfo.cpp
  X86SelectionDAGInfo.cpp
  X86Subtarget.cpp
  X86TargetMachine.cpp
  X86TargetObjectFile.cpp
  X86TargetTransformInfo.cpp
  X86VZeroUpper.cpp
  X86FixupLEAs.cpp
  X86WinEHState.cpp
  )

add_llvm_target(X86CodeGen ${sources})

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(InstPrinter)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
add_subdirectory(Utils)
