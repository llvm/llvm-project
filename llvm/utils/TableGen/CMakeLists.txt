set(LLVM_LINK_COMPONENTS Support)

add_tablegen(llvm-tblgen LLVM
  AsmMatcherEmitter.cpp
  AsmWriterEmitter.cpp
  AsmWriterInst.cpp
  Attributes.cpp
  CallingConvEmitter.cpp
  CodeEmitterGen.cpp
  CodeGenDAGPatterns.cpp
  CodeGenInstruction.cpp
  CodeGenMapTable.cpp
  CodeGenRegisters.cpp
  CodeGenSchedule.cpp
  CodeGenTarget.cpp
  DAGISelEmitter.cpp
  DAGISelMatcherEmitter.cpp
  DAGISelMatcherGen.cpp
  DAGISelMatcherOpt.cpp
  DAGISelMatcher.cpp
  DFAPacketizerEmitter.cpp
  DisassemblerEmitter.cpp
  FastISelEmitter.cpp
  FixedLenDecoderEmitter.cpp
  InstrInfoEmitter.cpp
  IntrinsicEmitter.cpp
  OptParserEmitter.cpp
  PseudoLoweringEmitter.cpp
  RegisterInfoEmitter.cpp
  SubtargetEmitter.cpp
  TableGen.cpp
  X86DisassemblerTables.cpp
  X86ModRMFilters.cpp
  X86RecognizableInstr.cpp
  CTagsEmitter.cpp
  )
