add_lldb_library(lldbUtility
  ConstString.cpp
  DataBufferHeap.cpp
  DataBufferLLVM.cpp
  DataEncoder.cpp
  DataExtractor.cpp
  Error.cpp
  JSON.cpp
  LLDBAssert.cpp
  Log.cpp
  Logging.cpp
  NameMatches.cpp
  Range.cpp
  RegularExpression.cpp
  SelectHelper.cpp
  SharingPtr.cpp
  Stream.cpp
  StreamString.cpp
  StringExtractor.cpp
  StringExtractorGDBRemote.cpp
  StringLexer.cpp
  TaskPool.cpp
  UriParser.cpp
  UUID.cpp
  VASprintf.cpp

  LINK_LIBS
  # lldbUtility cannot have any dependencies

  LINK_COMPONENTS
    Support
  )
