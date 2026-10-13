# orc-rt-connect-socket: the SimpleRemote protocol over a connected stream
# socket, and the "socket:adopt" connector for a socket this process was handed.
#
# Nothing here establishes a connection. Connecting and listening belong to the
# libraries for those mechanisms, built on this one, so that linking this
# library alone gives a process no way to reach a peer, or be reached by one.

# Socket support is per system only, so there are no portable sources.
set(ORC_RT_CONNECT_SOCKET_SOURCES
  )

set(ORC_RT_CONNECT_SOCKET_POSIX_SOURCES
  sys/posix/SocketConnector.cpp
  sys/posix/SocketHandle.cpp
  sys/posix/sps/SimpleRemoteCAOverSocket.cpp
)

# TODO: Windows has stubs only. The connector reports the socket transport as
# unimplemented, SocketHandle does not close the socket it owns, and there is no
# CA.
set(ORC_RT_CONNECT_SOCKET_WINDOWS_SOURCES
  sys/windows/SocketConnector.cpp
  sys/windows/SocketHandle.cpp
)

# Composed the way lib/bedrock/CMakeLists.txt composes its per-system sources;
# see the note there.
if (ORC_RT_TARGET_POSIX)
  list(APPEND ORC_RT_CONNECT_SOCKET_SOURCES
    ${ORC_RT_CONNECT_SOCKET_POSIX_SOURCES})
elseif (WIN32)
  list(APPEND ORC_RT_CONNECT_SOCKET_SOURCES
    ${ORC_RT_CONNECT_SOCKET_WINDOWS_SOURCES})
else()
  # With no portable sources there is no library to build.
  message(WARNING
    "orc-rt: no socket implementation for ${CMAKE_SYSTEM_NAME}; "
    "add an orc-rt/lib/connect-socket/sys/<system>/ directory and compose it "
    "above")
  return()
endif()

add_library(orc-rt-connect-socket-objects OBJECT
  ${ORC_RT_CONNECT_SOCKET_SOURCES})

set_target_properties(orc-rt-connect-socket-objects PROPERTIES
  POSITION_INDEPENDENT_CODE ${ORC_RT_ENABLE_PIC}
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

target_link_libraries(orc-rt-connect-socket-objects PUBLIC orc-rt-headers)

# Apply RTTI and exceptions compile flags
target_compile_options(orc-rt-connect-socket-objects
  PRIVATE ${ORC_RT_COMPILE_FLAGS})

# The shipped library. Support comes through orc-rt-connect's dependence on
# Bedrock; see lib/connect/CMakeLists.txt.
add_library(orc-rt-connect-socket ${ORC_RT_BEDROCK_LIBRARY_TYPE})

target_link_libraries(orc-rt-connect-socket
  PUBLIC orc-rt-headers orc-rt-connect)

target_link_libraries(orc-rt-connect-socket
  PRIVATE $<BUILD_INTERFACE:orc-rt-connect-socket-objects>)

# TODO: Use common runtimes infrastructure for output and install paths
install(TARGETS orc-rt-connect-socket
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          COMPONENT OrcRT_Development
  PUBLIC_HEADER DESTINATION include COMPONENT OrcRT_Development
)
