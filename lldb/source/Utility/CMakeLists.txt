add_lldb_library(lldbUtility
  ConstString.cpp
  Error.cpp
  JSON.cpp
  LLDBAssert.cpp
  NameMatches.cpp
  PseudoTerminal.cpp
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

  LINK_LIBS
    lldbHost

  LINK_COMPONENTS
    Support
  )
