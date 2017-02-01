add_lldb_library(lldbUtility
  JSON.cpp
  LLDBAssert.cpp
  ModuleCache.cpp
  NameMatches.cpp
  PseudoTerminal.cpp
  Range.cpp
  RegisterNumber.cpp
  SelectHelper.cpp
  SharingPtr.cpp
  StringExtractor.cpp
  StringExtractorGDBRemote.cpp
  StringLexer.cpp
  TaskPool.cpp
  UriParser.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbTarget

  LINK_COMPONENTS
    Support
  )
