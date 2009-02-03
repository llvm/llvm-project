add_executable(tblgen
  AsmWriterEmitter.cpp
  CallingConvEmitter.cpp
  CodeEmitterGen.cpp
  CodeGenDAGPatterns.cpp
  CodeGenInstruction.cpp
  CodeGenTarget.cpp
  DAGISelEmitter.cpp
  InstrEnumEmitter.cpp
  InstrInfoEmitter.cpp
  IntrinsicEmitter.cpp
  LLVMCConfigurationEmitter.cpp
  Record.cpp
  RegisterInfoEmitter.cpp
  SubtargetEmitter.cpp
  TGLexer.cpp
  TGParser.cpp
  TGValueTypes.cpp
  TableGen.cpp
  TableGenBackend.cpp
  FastISelEmitter.cpp
  )

target_link_libraries(tblgen LLVMSupport LLVMSystem)
if( MINGW )
  target_link_libraries(tblgen imagehlp psapi)
endif( MINGW )
