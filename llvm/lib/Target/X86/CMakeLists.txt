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
tablegen(LLVM X86GenEDInfo.inc -gen-enhanced-disassembly-info)
add_public_tablegen_target(X86CommonTableGen)

set(sources
  X86AsmPrinter.cpp
  X86COFFMachineModuleInfo.cpp
  X86CodeEmitter.cpp
  X86FastISel.cpp
  X86FloatingPoint.cpp
  X86FrameLowering.cpp
  X86ISelDAGToDAG.cpp
  X86ISelLowering.cpp
  X86InstrInfo.cpp
  X86JITInfo.cpp
  X86MCInstLower.cpp
  X86MachineFunctionInfo.cpp
  X86RegisterInfo.cpp
  X86SelectionDAGInfo.cpp
  X86Subtarget.cpp
  X86TargetMachine.cpp
  X86TargetObjectFile.cpp
  X86VZeroUpper.cpp
  )

if( CMAKE_CL_64 )
  # A workaround for a bug in cmake 2.8.3. See PR 8885.
  if( CMAKE_VERSION STREQUAL "2.8.3" )
    include(CMakeDetermineCompilerId)
  endif()
  # end of workaround.
  enable_language(ASM_MASM)
  ADD_CUSTOM_COMMAND(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/X86CompilationCallback_Win64.obj
    MAIN_DEPENDENCY X86CompilationCallback_Win64.asm
    COMMAND ${CMAKE_ASM_MASM_COMPILER} /Fo ${CMAKE_CURRENT_BINARY_DIR}/X86CompilationCallback_Win64.obj /c ${CMAKE_CURRENT_SOURCE_DIR}/X86CompilationCallback_Win64.asm
   )
   set(sources ${sources} ${CMAKE_CURRENT_BINARY_DIR}/X86CompilationCallback_Win64.obj)
endif()

add_llvm_target(X86CodeGen ${sources})

add_dependencies(LLVMX86CodeGen intrinsics_gen)

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(InstPrinter)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
add_subdirectory(Utils)
