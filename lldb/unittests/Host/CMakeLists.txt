add_lldb_unittest(HostTests
  FileSpecTest.cpp
  FileSystemTest.cpp
  SocketAddressTest.cpp
  SocketTest.cpp
  SymbolsTest.cpp

  LINK_LIBS
    lldbCore
    lldbHost
  )
