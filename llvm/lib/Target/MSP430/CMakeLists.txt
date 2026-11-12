add_llvm_component_group(MSP430)

set(LLVM_TARGET_DEFINITIONS MSP430.td)

llvm_tablegen(MSP430GenAsmMatcher.inc -gen-asm-matcher)
llvm_tablegen(MSP430GenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(MSP430GenCallingConv.inc -gen-callingconv)
llvm_tablegen(MSP430GenDAGISel.inc -gen-dag-isel)
llvm_tablegen(MSP430GenDisassemblerTables.inc -gen-disassembler)
llvm_tablegen(MSP430GenInstrInfo.inc -gen-instr-info)
llvm_tablegen(MSP430GenMCCodeEmitter.inc -gen-emitter)
llvm_tablegen(MSP430GenRegisterInfo.inc -gen-register-info)
llvm_tablegen(MSP430GenSDNodeInfo.inc -gen-sd-node-info)
llvm_tablegen(MSP430GenSubtargetInfo.inc -gen-subtarget)

set(LLVM_TARGET_DEFINITIONS MSP430Options.td)
llvm_tablegen(MSP430Options.inc -gen-opt-parser-defs)

add_public_tablegen_target(MSP430CommonTableGen)

add_llvm_target(MSP430CodeGen
  MSP430BranchSelector.cpp
  MSP430CodeGenPassBuilder.cpp
  MSP430ISelDAGToDAG.cpp
  MSP430ISelLowering.cpp
  MSP430InstrInfo.cpp
  MSP430FrameLowering.cpp
  MSP430MachineFunctionInfo.cpp
  MSP430RegisterInfo.cpp
  MSP430SelectionDAGInfo.cpp
  MSP430Subtarget.cpp
  MSP430TargetMachine.cpp
  MSP430AsmPrinter.cpp
  MSP430MCInstLower.cpp

  LINK_COMPONENTS
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  MC
  MSP430Desc
  MSP430Info
  Option
  Passes
  SelectionDAG
  Support
  Target
  TargetParser

  ADD_TO_COMPONENT
  MSP430
  )

add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
