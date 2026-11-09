add_llvm_component_group(SystemZ HAS_JIT)

set(LLVM_TARGET_DEFINITIONS SystemZ.td)

llvm_tablegen(SystemZGenAsmMatcher.inc -gen-asm-matcher)
llvm_tablegen(SystemZGenGNUAsmWriter.inc -gen-asm-writer)
llvm_tablegen(SystemZGenHLASMAsmWriter.inc -gen-asm-writer -asmwriternum=1)
llvm_tablegen(SystemZGenCallingConv.inc -gen-callingconv)
llvm_tablegen(SystemZGenDAGISel.inc -gen-dag-isel)
llvm_tablegen(SystemZGenDisassemblerTables.inc -gen-disassembler)
llvm_tablegen(SystemZGenInsnMatchTable.inc -gen-insn-directive-match-table)
llvm_tablegen(SystemZGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(SystemZGenMCCodeEmitter.inc -gen-emitter)
llvm_tablegen(SystemZGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(SystemZGenSDNodeInfo.inc -gen-sd-node-info)
llvm_tablegen(SystemZGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(SystemZCommonTableGen)

add_llvm_target(SystemZCodeGen
  SystemZAsmPrinter.cpp
  SystemZAlignGlobals.cpp
  SystemZCallingConv.cpp
  SystemZConstantPoolValue.cpp
  SystemZCopyPhysRegs.cpp
  SystemZELFAsmPrinter.cpp
  SystemZElimCompare.cpp
  SystemZFrameLowering.cpp
  SystemZHazardRecognizer.cpp
  SystemZISelDAGToDAG.cpp
  SystemZISelLowering.cpp
  SystemZInstrInfo.cpp
  SystemZLDCleanup.cpp
  SystemZLongBranch.cpp
  SystemZMachineFunctionInfo.cpp
  SystemZMachineScheduler.cpp
  SystemZMCInstLower.cpp
  SystemZPostRewrite.cpp
  SystemZRegisterInfo.cpp
  SystemZSelectionDAGInfo.cpp
  SystemZShortenInst.cpp
  SystemZSubtarget.cpp
  SystemZTargetMachine.cpp
  SystemZTargetObjectFile.cpp
  SystemZTargetTransformInfo.cpp
  SystemZTDC.cpp
  SystemZXPLINKAsmPrinter.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  MC
  Scalar
  SelectionDAG
  Support
  SystemZDesc
  SystemZInfo
  Target
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT
  SystemZ
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
