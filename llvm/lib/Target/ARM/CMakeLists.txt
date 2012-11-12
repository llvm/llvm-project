set(LLVM_TARGET_DEFINITIONS ARM.td)

tablegen(LLVM ARMGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM ARMGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM ARMGenCodeEmitter.inc -gen-emitter)
tablegen(LLVM ARMGenMCCodeEmitter.inc -gen-emitter -mc-emitter)
tablegen(LLVM ARMGenMCPseudoLowering.inc -gen-pseudo-lowering)
tablegen(LLVM ARMGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM ARMGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM ARMGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM ARMGenFastISel.inc -gen-fast-isel)
tablegen(LLVM ARMGenCallingConv.inc -gen-callingconv)
tablegen(LLVM ARMGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM ARMGenEDInfo.inc -gen-enhanced-disassembly-info)
tablegen(LLVM ARMGenDisassemblerTables.inc -gen-disassembler)
add_public_tablegen_target(ARMCommonTableGen)

add_llvm_target(ARMCodeGen
  ARMAsmPrinter.cpp
  ARMBaseInstrInfo.cpp
  ARMBaseRegisterInfo.cpp
  ARMCodeEmitter.cpp
  ARMConstantIslandPass.cpp
  ARMConstantPoolValue.cpp
  ARMExpandPseudoInsts.cpp
  ARMFastISel.cpp
  ARMFrameLowering.cpp
  ARMHazardRecognizer.cpp
  ARMISelDAGToDAG.cpp
  ARMISelLowering.cpp
  ARMInstrInfo.cpp
  ARMJITInfo.cpp
  ARMLoadStoreOptimizer.cpp
  ARMMCInstLower.cpp
  ARMMachineFunctionInfo.cpp
  ARMRegisterInfo.cpp
  ARMSelectionDAGInfo.cpp
  ARMSubtarget.cpp
  ARMTargetMachine.cpp
  ARMTargetObjectFile.cpp
  MLxExpansionPass.cpp
  Thumb1FrameLowering.cpp
  Thumb1InstrInfo.cpp
  Thumb1RegisterInfo.cpp
  Thumb2ITBlockPass.cpp
  Thumb2InstrInfo.cpp
  Thumb2RegisterInfo.cpp
  Thumb2SizeReduction.cpp
  )

add_dependencies(LLVMARMCodeGen intrinsics_gen)

# workaround for hanging compilation on MSVC9, 10
if( MSVC_VERSION EQUAL 1600 OR MSVC_VERSION EQUAL 1500 )
set_property(
  SOURCE ARMISelLowering.cpp
  PROPERTY COMPILE_FLAGS "/Od"
  )
endif()

add_subdirectory(TargetInfo)
add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(InstPrinter)
add_subdirectory(MCTargetDesc)
