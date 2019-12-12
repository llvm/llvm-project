macro(add_host_subdirectory group)
  list(APPEND HOST_SOURCES ${ARGN})
  source_group(${group} FILES ${ARGN})
endmacro()

# Removes all module flags from the current CMAKE_CXX_FLAGS. Used for
# the Objective-C++ code in lldb which we don't want to build with modules.
# Reasons for this are that modules with Objective-C++ would require that
# all LLVM/Clang modules are Objective-C++ compatible (which they are likely
# not) and we would have rebuild a second set of modules just for the few
# Objective-C++ files in lldb (which slows down the build process).
macro(remove_module_flags)
  string(REGEX REPLACE "-fmodules-cache-path=[^ ]+" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REGEX REPLACE "-fmodules-local-submodule-visibility" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REGEX REPLACE "-fmodules" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REGEX REPLACE "-gmodules" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REGEX REPLACE "-fcxx-modules" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endmacro()

add_host_subdirectory(common
  common/FileAction.cpp
  common/FileCache.cpp
  common/File.cpp
  common/FileSystem.cpp
  common/GetOptInc.cpp
  common/Host.cpp
  common/HostInfoBase.cpp
  common/HostNativeThreadBase.cpp
  common/HostProcess.cpp
  common/HostThread.cpp
  common/LockFileBase.cpp
  common/LZMA.cpp
  common/MainLoop.cpp
  common/MonitoringProcessLauncher.cpp
  common/NativeProcessProtocol.cpp
  common/NativeRegisterContext.cpp
  common/NativeThreadProtocol.cpp
  common/NativeWatchpointList.cpp
  common/OptionParser.cpp
  common/PipeBase.cpp
  common/ProcessLaunchInfo.cpp
  common/ProcessRunLock.cpp
  common/PseudoTerminal.cpp
  common/SocketAddress.cpp
  common/Socket.cpp
  common/StringConvert.cpp
  common/TaskPool.cpp
  common/TCPSocket.cpp
  common/Terminal.cpp
  common/ThreadLauncher.cpp
  common/UDPSocket.cpp
  common/XML.cpp
  )

if (LLDB_ENABLE_LIBEDIT)
  add_host_subdirectory(common
    common/Editline.cpp
    )
endif()

add_host_subdirectory(posix
  posix/ConnectionFileDescriptorPosix.cpp
  )

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
  add_host_subdirectory(windows
    windows/ConnectionGenericFileWindows.cpp
    windows/EditLineWin.cpp
    windows/FileSystem.cpp
    windows/Host.cpp
    windows/HostInfoWindows.cpp
    windows/HostProcessWindows.cpp
    windows/HostThreadWindows.cpp
    windows/LockFileWindows.cpp
    windows/PipeWindows.cpp
    windows/ProcessLauncherWindows.cpp
    windows/ProcessRunLock.cpp
    windows/Windows.cpp
    )
else()
  add_host_subdirectory(posix
    posix/DomainSocket.cpp
    posix/FileSystemPosix.cpp
    posix/HostInfoPosix.cpp
    posix/HostProcessPosix.cpp
    posix/HostThreadPosix.cpp
    posix/LockFilePosix.cpp
    posix/PipePosix.cpp
    posix/ProcessLauncherPosixFork.cpp
    )

  if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    include_directories(SYSTEM ${LIBXML2_INCLUDE_DIR})
    add_subdirectory(macosx/objcxx)
    set(LLDBObjCLibs lldbHostMacOSXObjCXX)
    add_host_subdirectory(macosx
      macosx/cfcpp/CFCBundle.cpp
      macosx/cfcpp/CFCData.cpp
      macosx/cfcpp/CFCMutableArray.cpp
      macosx/cfcpp/CFCMutableDictionary.cpp
      macosx/cfcpp/CFCMutableSet.cpp
      macosx/cfcpp/CFCString.cpp
      )
    if(IOS)
      set_property(SOURCE macosx/Host.mm APPEND PROPERTY
               COMPILE_DEFINITIONS "NO_XPC_SERVICES=1")
    endif()


  elseif (CMAKE_SYSTEM_NAME MATCHES "Linux|Android")
    add_host_subdirectory(linux
      linux/AbstractSocket.cpp
      linux/Host.cpp
      linux/HostInfoLinux.cpp
      linux/LibcGlue.cpp
      linux/Support.cpp
      )
    if (CMAKE_SYSTEM_NAME MATCHES "Android")
      add_host_subdirectory(android
        android/HostInfoAndroid.cpp
        android/LibcGlue.cpp
        )
    endif()
  elseif (CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
    add_host_subdirectory(freebsd
      freebsd/Host.cpp
      freebsd/HostInfoFreeBSD.cpp
      )

  elseif (CMAKE_SYSTEM_NAME MATCHES "NetBSD")
    add_host_subdirectory(netbsd
      netbsd/Host.cpp
      netbsd/HostInfoNetBSD.cpp
      )

  elseif (CMAKE_SYSTEM_NAME MATCHES "OpenBSD")
    add_host_subdirectory(openbsd
      openbsd/Host.cpp
      openbsd/HostInfoOpenBSD.cpp
      )
  endif()
endif()

set(EXTRA_LIBS)
if (CMAKE_SYSTEM_NAME MATCHES "NetBSD")
  list(APPEND EXTRA_LIBS kvm)
endif ()
if (APPLE)
  list(APPEND EXTRA_LIBS xml2)
else ()
  if (LIBXML2_FOUND)
    list(APPEND EXTRA_LIBS ${LIBXML2_LIBRARIES})
  endif()
endif ()
if (HAVE_LIBDL)
  list(APPEND EXTRA_LIBS ${CMAKE_DL_LIBS})
endif()
if (LLDB_ENABLE_LIBEDIT)
  list(APPEND EXTRA_LIBS ${libedit_LIBRARIES})
endif()
if (LLDB_ENABLE_LZMA)
  list(APPEND EXTRA_LIBS ${LIBLZMA_LIBRARIES})
endif()

if (LLDB_ENABLE_LIBEDIT)
  list(APPEND LLDB_LIBEDIT_LIBS ${libedit_LIBRARIES})
  if (LLVM_BUILD_STATIC)
    list(APPEND LLDB_SYSTEM_LIBS gpm)
  endif()
endif()

add_lldb_library(lldbHost
  ${HOST_SOURCES}

  LINK_LIBS
    lldbUtility
    ${EXTRA_LIBS}
    ${LLDBObjCLibs}
    ${LLDB_LIBEDIT_LIBS}

  LINK_COMPONENTS
    Object
    Support
  )

if (LLDB_ENABLE_LIBEDIT)
  target_include_directories(lldbHost PUBLIC ${libedit_INCLUDE_DIRS})
endif()
