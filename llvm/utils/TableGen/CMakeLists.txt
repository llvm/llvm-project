add_subdirectory(GlobalISel)

add_llvm_library(LLVMTableGenCommon STATIC OBJECT EXCLUDE_FROM_ALL
  Attributes.cpp
  CodeGenIntrinsics.cpp
  DirectiveEmitter.cpp
  IntrinsicEmitter.cpp
  RISCVTargetDefEmitter.cpp
  SDNodeProperties.cpp
  VTEmitter.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_COMPONENTS
  Support
  TableGen
  )
set_target_properties(LLVMTableGenCommon PROPERTIES FOLDER "Tablegenning")

set(LLVM_LINK_COMPONENTS Support)

add_tablegen(llvm-min-tblgen LLVM_HEADERS
  TableGen.cpp
  $<TARGET_OBJECTS:obj.LLVMTableGenCommon>
  PARTIAL_SOURCES_INTENDED
  )
set_target_properties(llvm-min-tblgen PROPERTIES FOLDER "Tablegenning")

set(LLVM_LINK_COMPONENTS
  CodeGenTypes
  Support
  )

add_tablegen(llvm-tblgen LLVM
  DESTINATION "${LLVM_TOOLS_INSTALL_DIR}"
  EXPORT LLVM
  AsmMatcherEmitter.cpp
  AsmWriterEmitter.cpp
  AsmWriterInst.cpp
  CTagsEmitter.cpp
  CallingConvEmitter.cpp
  CodeEmitterGen.cpp
  CodeGenDAGPatterns.cpp
  CodeGenHwModes.cpp
  CodeGenInstAlias.cpp
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
  DisassemblerEmitter.cpp
  DXILEmitter.cpp
  ExegesisEmitter.cpp
  FastISelEmitter.cpp
  GICombinerEmitter.cpp
  GlobalISelEmitter.cpp
  InfoByHwMode.cpp
  InstrInfoEmitter.cpp
  InstrDocsEmitter.cpp
  OptEmitter.cpp
  OptParserEmitter.cpp
  OptRSTEmitter.cpp
  PredicateExpander.cpp
  PseudoLoweringEmitter.cpp
  CompressInstEmitter.cpp
  RegisterBankEmitter.cpp
  RegisterInfoEmitter.cpp
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
  $<TARGET_OBJECTS:obj.LLVMTableGenCommon>

  DEPENDS
  intrinsics_gen # via llvm-min-tablegen
  )
target_link_libraries(llvm-tblgen PRIVATE LLVMTableGenGlobalISel)
set_target_properties(llvm-tblgen PROPERTIES FOLDER "Tablegenning")
