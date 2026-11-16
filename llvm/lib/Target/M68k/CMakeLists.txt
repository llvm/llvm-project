add_llvm_component_group(M68k)

set(LLVM_TARGET_DEFINITIONS M68k.td)

llvm_tablegen(M68kGenGlobalISel.inc       -gen-global-isel)
llvm_tablegen(M68kGenRegisterInfo.inc     -gen-register-info)
llvm_tablegen(M68kGenRegisterBank.inc     -gen-register-bank)
llvm_tablegen(M68kGenInstrInfo.inc        -gen-instr-info)
llvm_tablegen(M68kGenSDNodeInfo.inc       -gen-sd-node-info)
llvm_tablegen(M68kGenSubtargetInfo.inc    -gen-subtarget)
llvm_tablegen(M68kGenMCCodeEmitter.inc    -gen-emitter)
llvm_tablegen(M68kGenMCPseudoLowering.inc -gen-pseudo-lowering)
llvm_tablegen(M68kGenDAGISel.inc          -gen-dag-isel)
llvm_tablegen(M68kGenCallingConv.inc      -gen-callingconv)
llvm_tablegen(M68kGenAsmWriter.inc        -gen-asm-writer)
llvm_tablegen(M68kGenAsmMatcher.inc       -gen-asm-matcher)
llvm_tablegen(M68kGenDisassemblerTables.inc -gen-disassembler)

add_public_tablegen_target(M68kCommonTableGen)

add_llvm_target(M68kCodeGen
  GISel/M68kCallLowering.cpp
  GISel/M68kInstructionSelector.cpp
  GISel/M68kLegalizerInfo.cpp
  GISel/M68kRegisterBankInfo.cpp
  M68kAsmPrinter.cpp
  M68kCollapseMOVEMPass.cpp
  M68kExpandPseudo.cpp
  M68kFrameLowering.cpp
  M68kInstrInfo.cpp
  M68kISelLowering.cpp
  M68kISelDAGToDAG.cpp
  M68kMachineFunction.cpp
  M68kMCInstLower.cpp
  M68kRegisterInfo.cpp
  M68kSelectionDAGInfo.cpp
  M68kSubtarget.cpp
  M68kTargetMachine.cpp
  M68kTargetObjectFile.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  GlobalISel
  M68kDesc
  M68kInfo
  MC
  SelectionDAG
  Support
  Target
  TargetParser

  ADD_TO_COMPONENT
  M68k
  )

add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
