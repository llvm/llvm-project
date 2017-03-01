add_lldb_library(lldbUtility
  ConstString.cpp
  Error.cpp
  JSON.cpp
  LLDBAssert.cpp
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
  VASprintf.cpp

  LINK_LIBS
  # lldbUtility cannot have any dependencies

  LINK_COMPONENTS
    Support
  )
