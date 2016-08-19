include_directories(${CMAKE_CURRENT_BINARY_DIR}/..)
include_directories(${LLDB_SOURCE_DIR}/source)
include_directories(MacOSX/DarwinLog)

if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  include_directories(MacOSX)
  #include_directories(${CMAKE_CURRENT_BINARY_DIR}/MacOSX)

  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -Wl,-sectcreate,__TEXT,__info_plist,${CMAKE_CURRENT_SOURCE_DIR}/../resources/lldb-debugserver-Info.plist")
endif()

check_cxx_compiler_flag("-Wno-gnu-zero-variadic-macro-arguments"
                        CXX_SUPPORTS_NO_GNU_ZERO_VARIADIC_MACRO_ARGUMENTS)
if (CXX_SUPPORTS_NO_GNU_ZERO_VARIADIC_MACRO_ARGUMENTS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-gnu-zero-variadic-macro-arguments")
endif ()

check_cxx_compiler_flag("-Wno-zero-length-array"
                        CXX_SUPPORTS_NO_ZERO_LENGTH_ARRAY)
if (CXX_SUPPORTS_NO_ZERO_LENGTH_ARRAY)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-zero-length-array")
endif ()

check_cxx_compiler_flag("-Wno-extended-offsetof"
                        CXX_SUPPORTS_NO_EXTENDED_OFFSETOF)
if (CXX_SUPPORTS_NO_EXTENDED_OFFSETOF)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-extended-offsetof")
endif ()

if (NOT CMAKE_SYSTEM_NAME MATCHES "Darwin")
  add_definitions(
    -DDEBUGSERVER_VERSION_STR="${LLDB_VERSION}"
    )
endif ()

add_library(lldbDebugserverCommon
  debugserver.cpp
  DNBArch.cpp
  DNBBreakpoint.cpp
  DNB.cpp
  DNBDataRef.cpp
  DNBError.cpp
  DNBLog.cpp
  DNBRegisterInfo.cpp
  DNBThreadResumeActions.cpp
  JSON.cpp
  # JSON reader depends on the following LLDB-common files
  ${LLDB_SOURCE_DIR}/source/Host/common/StringConvert.cpp
  ${LLDB_SOURCE_DIR}/source/Utility/StringExtractor.cpp
  # end JSON reader dependencies
  libdebugserver.cpp
  PseudoTerminal.cpp
  PThreadEvent.cpp
  PThreadMutex.cpp
  RNBContext.cpp
  RNBRemote.cpp
  RNBServices.cpp
  RNBSocket.cpp
  SysSignal.cpp
  TTYState.cpp
  )

if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  find_library(COCOA_LIBRARY Cocoa)
  target_link_libraries(lldbDebugserverCommon ${COCOA_LIBRARY})
  add_subdirectory(MacOSX)
endif()

