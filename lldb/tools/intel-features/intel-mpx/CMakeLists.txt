add_lldb_library(lldbIntelMPX
  cli-wrapper-mpxtable.cpp

  LINK_COMPONENTS
    Support
    TargetParser
  LINK_LIBS
    liblldb
  )
