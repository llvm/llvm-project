# Basic utilities which is the strict minimum needed to build
# llvm-min-tblgen.
add_subdirectory(Basic)
# Common utilities are all of the reusable components and helper
# code needed by the backends.
add_subdirectory(Common)

set(LLVM_LINK_COMPONENTS Support)

# llvm-min-tablegen only contains a subset of backends necessary to
# build llvm/include. It must not depend on TableGenCommon, as
# TableGenCommon depends on this already to generate things such as
# ValueType definitions.
# Sources included in both, llvm-min-tblgen and llvm-tblgen, must be included
# into LLVMTableGenBasic to avoid redundant compilation and problems with build
# caches.
# At least one source file must be included directly to avoid CMake problems.
# E.g. CMake derives which linker to use from the types of sources added.
add_tablegen(llvm-min-tblgen LLVM_HEADERS
  llvm-min-tblgen.cpp
  $<TARGET_OBJECTS:obj.LLVMTableGenBasic>

  PARTIAL_SOURCES_INTENDED
  )

set(LLVM_LINK_COMPONENTS
  CodeGenTypes
  Support
  )

add_tablegen(llvm-tblgen LLVM
  DESTINATION "${LLVM_TOOLS_INSTALL_DIR}"
  EXPORT LLVM
  AsmMatcherEmitter.cpp
  AsmWriterEmitter.cpp
  CallingConvEmitter.cpp
  CodeEmitterGen.cpp
  CodeGenMapTable.cpp
  CompressInstEmitter.cpp
  CTagsEmitter.cpp
  DAGISelEmitter.cpp
  DAGISelMatcherEmitter.cpp
  DAGISelMatcherGen.cpp
  DAGISelMatcherOpt.cpp
  DecoderEmitter.cpp
  DFAEmitter.cpp
  DFAPacketizerEmitter.cpp
  DisassemblerEmitter.cpp
  DXILEmitter.cpp
  ExegesisEmitter.cpp
  FastISelEmitter.cpp
  GlobalISelCombinerEmitter.cpp
  GlobalISelEmitter.cpp
  InstrDocsEmitter.cpp
  InstrInfoEmitter.cpp
  llvm-tblgen.cpp
  MacroFusionPredicatorEmitter.cpp
  OptionParserEmitter.cpp
  OptionRSTEmitter.cpp
  PseudoLoweringEmitter.cpp
  RegisterBankEmitter.cpp
  RegisterInfoEmitter.cpp
  SDNodeInfoEmitter.cpp
  SearchableTableEmitter.cpp
  SubtargetEmitter.cpp
  WebAssemblyDisassemblerEmitter.cpp
  X86InstrMappingEmitter.cpp
  X86DisassemblerTables.cpp
  X86FoldTablesEmitter.cpp
  X86MnemonicTables.cpp
  X86ModRMFilters.cpp
  X86RecognizableInstr.cpp
  $<TARGET_OBJECTS:obj.LLVMTableGenBasic>
  $<TARGET_OBJECTS:obj.LLVMTableGenCommon>

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  intrinsics_gen # via llvm-min-tablegen
  )
