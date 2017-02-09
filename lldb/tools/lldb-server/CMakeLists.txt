if ( CMAKE_SYSTEM_NAME MATCHES "Linux" )
include_directories(
  ../../../../llvm/include
  ../../source/Plugins/Process/Linux
  ../../source/Plugins/Process/POSIX
  )
endif ()

if ( CMAKE_SYSTEM_NAME MATCHES "FreeBSD" )
include_directories(
  ../../../../llvm/include
  ../../source/Plugins/Process/FreeBSD
  ../../source/Plugins/Process/POSIX
  )
endif ()

if ( CMAKE_SYSTEM_NAME MATCHES "NetBSD" )
include_directories(
  ../../../../llvm/include
  ../../source/Plugins/Process/POSIX
  )
endif ()

include_directories(../../source)

set(LLDB_SYSTEM_LIBS)
if (NOT LLDB_DISABLE_LIBEDIT)
  list(APPEND LLDB_SYSTEM_LIBS edit)
endif()
if (NOT LLDB_DISABLE_CURSES)
  list(APPEND LLDB_SYSTEM_LIBS ${CURSES_LIBRARIES})
  if(LLVM_ENABLE_TERMINFO AND HAVE_TERMINFO)
    list(APPEND LLDB_SYSTEM_LIBS ${TERMINFO_LIBS})
  endif()
endif()

if (NOT HAVE_CXX_ATOMICS64_WITHOUT_LIB )
    list(APPEND LLDB_SYSTEM_LIBS atomic)
endif()

# On FreeBSD/NetBSD backtrace() is provided by libexecinfo, not libc.
if (CMAKE_SYSTEM_NAME MATCHES "FreeBSD" OR CMAKE_SYSTEM_NAME MATCHES "NetBSD")
  list(APPEND LLDB_SYSTEM_LIBS execinfo)
endif()

if (NOT LLDB_DISABLE_PYTHON AND NOT LLVM_BUILD_STATIC)
  list(APPEND LLDB_SYSTEM_LIBS ${PYTHON_LIBRARIES})
endif()

list(APPEND LLDB_SYSTEM_LIBS ${system_libs})

if (LLVM_BUILD_STATIC)
  if (NOT LLDB_DISABLE_PYTHON)
    list(APPEND LLDB_SYSTEM_LIBS python2.7 util)
  endif()
  if (NOT LLDB_DISABLE_CURSES)
    list(APPEND LLDB_SYSTEM_LIBS gpm)
  endif()
endif()

add_lldb_tool(lldb-server INCLUDE_IN_FRAMEWORK
    Acceptor.cpp
    lldb-gdbserver.cpp
    lldb-platform.cpp
    lldb-server.cpp
    LLDBServerUtilities.cpp

    LINK_LIBS
      lldbBase
      lldbCore
      lldbHost
      lldbInitialization
      lldbInterpreter
      ${EXTRA_LLDB_LIBS}
      ${LLDB_SYSTEM_LIBS}

    LINK_COMPONENTS
      Support
)

target_link_libraries(lldb-server ${LLDB_SYSTEM_LIBS})
