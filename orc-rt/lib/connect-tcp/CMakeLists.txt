# orc-rt-connect-tcp: primitives for establishing TCP connections, as
# connected sockets for orc-rt-connect-socket to carry.
#
# A process that links this library can establish its own connections, and so
# take code to run from whatever it reaches or whatever reaches it. Keeping that
# in its own library makes it a decision made at link time, and one that can be
# checked from a binary's link line.
#
# Each action gets its own translation unit, so a process that only connects
# does not link the code to listen.

# The connectors are portable: they reach the system through the primitives in
# connect-tcp/TCP.h, which have an implementation per system.
set(ORC_RT_CONNECT_TCP_SOURCES
  TCPConnectConnector.cpp
  TCPConnectorHelpers.cpp
  TCPListenConnector.cpp
)

set(ORC_RT_CONNECT_TCP_POSIX_SOURCES
  sys/posix/TCPConnect.cpp
  sys/posix/TCPListen.cpp
  sys/posix/TCPSocket.cpp
)

# TODO: Windows has stubs only, which report TCP as not yet supported.
set(ORC_RT_CONNECT_TCP_WINDOWS_SOURCES
  sys/windows/TCPConnect.cpp
  sys/windows/TCPListen.cpp
)

# Composed the way lib/bedrock/CMakeLists.txt composes its per-system sources;
# see the note there.
if (ORC_RT_TARGET_POSIX)
  list(APPEND ORC_RT_CONNECT_TCP_SOURCES ${ORC_RT_CONNECT_TCP_POSIX_SOURCES})
elseif (WIN32)
  list(APPEND ORC_RT_CONNECT_TCP_SOURCES ${ORC_RT_CONNECT_TCP_WINDOWS_SOURCES})
else()
  message(WARNING
    "orc-rt: no TCP implementation for ${CMAKE_SYSTEM_NAME}; "
    "add an orc-rt/lib/connect-tcp/sys/<system>/ directory and compose it "
    "above")
endif()

add_library(orc-rt-connect-tcp-objects OBJECT ${ORC_RT_CONNECT_TCP_SOURCES})

set_target_properties(orc-rt-connect-tcp-objects PROPERTIES
  POSITION_INDEPENDENT_CODE ${ORC_RT_ENABLE_PIC}
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

target_link_libraries(orc-rt-connect-tcp-objects PUBLIC orc-rt-headers)

# Apply RTTI and exceptions compile flags
target_compile_options(orc-rt-connect-tcp-objects
  PRIVATE ${ORC_RT_COMPILE_FLAGS})

# The shipped library. Support comes through orc-rt-connect's dependence on
# Bedrock; see lib/connect/CMakeLists.txt.
add_library(orc-rt-connect-tcp ${ORC_RT_BEDROCK_LIBRARY_TYPE})

target_link_libraries(orc-rt-connect-tcp
  PUBLIC orc-rt-headers orc-rt-connect-socket
  PRIVATE $<BUILD_INTERFACE:orc-rt-connect-tcp-objects>)

# TODO: Use common runtimes infrastructure for output and install paths
install(TARGETS orc-rt-connect-tcp
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          COMPONENT OrcRT_Development
  PUBLIC_HEADER DESTINATION include COMPONENT OrcRT_Development
)
