add_llvm_library(LLVMDebugInfoCodeView
  FieldListRecordBuilder.cpp
  Line.cpp
  ListRecordBuilder.cpp
  MemoryTypeTableBuilder.cpp
  MethodListRecordBuilder.cpp
  TypeDumper.cpp
  TypeRecordBuilder.cpp
  TypeTableBuilder.cpp
  TypeStream.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/DebugInfo/CodeView
  )
