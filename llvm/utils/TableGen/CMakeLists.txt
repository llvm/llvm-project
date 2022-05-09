add_subdirectory(GlobalISel)

set(LLVM_LINK_COMPONENTS Support)

add_tablegen(llvm-tblgen LLVM
  AsmMatcherEmitter.cpp
  AsmWriterEmitter.cpp
  AsmWriterInst.cpp
  Attributes.cpp
  CallingConvEmitter.cpp
  CodeBeadsGen.cpp
  CodeEmitterGen.cpp
  CodeGenDAGPatterns.cpp
  CodeGenHwModes.cpp
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
  DecoderEmitter.cpp
  DFAEmitter.cpp
  DFAPacketizerEmitter.cpp
  DirectiveEmitter.cpp
  DisassemblerEmitter.cpp
  ExegesisEmitter.cpp
  FastISelEmitter.cpp
  GICombinerEmitter.cpp
  GlobalISelEmitter.cpp
  InfoByHwMode.cpp
  InstrInfoEmitter.cpp
  InstrDocsEmitter.cpp
  IntrinsicEmitter.cpp
  OptEmitter.cpp
  OptParserEmitter.cpp
  OptRSTEmitter.cpp
  PredicateExpander.cpp
  PseudoLoweringEmitter.cpp
  CompressInstEmitter.cpp
  RegisterBankEmitter.cpp
  RegisterInfoEmitter.cpp
  SDNodeProperties.cpp
  SearchableTableEmitter.cpp
  SubtargetEmitter.cpp
  SubtargetFeatureInfo.cpp
  TableGen.cpp
  Types.cpp
  VarLenCodeEmitterGen.cpp
  X86DisassemblerTables.cpp
  X86EVEX2VEXTablesEmitter.cpp
  X86FoldTablesEmitter.cpp
  X86MnemonicTables.cpp
  X86ModRMFilters.cpp
  X86RecognizableInstr.cpp
  WebAssemblyDisassemblerEmitter.cpp
  CTagsEmitter.cpp
  )
target_link_libraries(llvm-tblgen PRIVATE LLVMTableGenGlobalISel)
set_target_properties(llvm-tblgen PROPERTIES FOLDER "Tablegenning")
