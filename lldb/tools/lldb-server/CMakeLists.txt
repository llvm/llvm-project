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


set( LLDB_USED_LIBS
  lldbBase
  lldbBreakpoint
  lldbCommands
  lldbDataFormatters
  lldbHost
  lldbCore
  lldbExpression
  lldbInitialization
  lldbInterpreter
  lldbSymbol
  lldbTarget
  lldbUtility

  # Plugins
  lldbPluginDisassemblerLLVM
  lldbPluginSymbolFileDWARF
  lldbPluginSymbolFilePDB
  lldbPluginSymbolFileSymtab
  lldbPluginDynamicLoaderPosixDYLD

  lldbPluginCPlusPlusLanguage
  lldbPluginGoLanguage
  lldbPluginJavaLanguage
  lldbPluginObjCLanguage
  lldbPluginObjCPlusPlusLanguage
  lldbPluginOCamlLanguage

  lldbPluginObjectFileELF
  lldbPluginObjectFileJIT
  lldbPluginSymbolVendorELF
  lldbPluginPlatformPOSIX
  lldbPluginObjectContainerBSDArchive
  lldbPluginObjectContainerMachOArchive
  lldbPluginProcessGDBRemote
  lldbPluginProcessUtility
  lldbPluginObjectContainerMachOArchive
  lldbPluginObjectContainerBSDArchive
  lldbPluginPlatformMacOSX
  lldbPluginUnwindAssemblyInstEmulation
  lldbPluginUnwindAssemblyX86
  lldbPluginAppleObjCRuntime
  lldbPluginCXXItaniumABI
  lldbPluginInstructionARM
  lldbPluginInstructionARM64
  lldbPluginInstructionMIPS
  lldbPluginInstructionMIPS64
  lldbPluginObjectFilePECOFF
  lldbPluginProcessElfCore
  lldbPluginExpressionParserClang
  lldbPluginExpressionParserGo
  )

# Linux-only libraries
if ( CMAKE_SYSTEM_NAME MATCHES "Linux" )
  list(APPEND LLDB_USED_LIBS
    lldbPluginProcessLinux
    lldbPluginProcessPOSIX
   )
endif ()

# Darwin-only libraries
if ( CMAKE_SYSTEM_NAME MATCHES "Darwin" )
  list(APPEND LLDB_USED_LIBS
    lldbPluginObjectFileMachO
    )
endif()

set( CLANG_USED_LIBS
  clangAnalysis
  clangAST
  clangBasic
  clangCodeGen
  clangDriver
  clangEdit
  clangFrontend
  clangLex
  clangParse
  clangRewrite
  clangRewriteFrontend
  clangSema
  clangSerialization
  )

set(LLDB_SYSTEM_LIBS)
if (NOT CMAKE_SYSTEM_NAME MATCHES "Windows" AND NOT __ANDROID_NDK__)
  if (NOT LLDB_DISABLE_LIBEDIT)
    list(APPEND LLDB_SYSTEM_LIBS edit)
  endif()
  if (NOT LLDB_DISABLE_CURSES)
    list(APPEND LLDB_SYSTEM_LIBS ${CURSES_LIBRARIES})
    if(LLVM_ENABLE_TERMINFO AND HAVE_TERMINFO)
      list(APPEND LLDB_SYSTEM_LIBS ${TERMINFO_LIBS})
    endif()
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

set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  interpreter
  asmparser
  bitreader
  bitwriter
  codegen
  demangle
  ipo
  selectiondag
  bitreader
  mc
  mcjit
  core
  mcdisassembler
  executionengine
  runtimedyld
  option
  support
  coverage
  target
  )

add_lldb_executable(lldb-server INCLUDE_IN_FRAMEWORK
    Acceptor.cpp
    lldb-gdbserver.cpp
    lldb-platform.cpp
    lldb-server.cpp
    LLDBServerUtilities.cpp
)

# The Darwin linker doesn't understand --start-group/--end-group.
if (LLDB_LINKER_SUPPORTS_GROUPS)
  target_link_libraries(lldb-server
                        -Wl,--start-group ${LLDB_USED_LIBS} -Wl,--end-group)
  target_link_libraries(lldb-server
                        -Wl,--start-group ${CLANG_USED_LIBS} -Wl,--end-group)
else()
  target_link_libraries(lldb-server ${LLDB_USED_LIBS})
  target_link_libraries(lldb-server ${CLANG_USED_LIBS})
endif()
llvm_config(lldb-server ${LLVM_LINK_COMPONENTS})

target_link_libraries(lldb-server ${LLDB_SYSTEM_LIBS})

set_target_properties(lldb-server PROPERTIES VERSION ${LLDB_VERSION})

install(TARGETS lldb-server
  RUNTIME DESTINATION bin)
