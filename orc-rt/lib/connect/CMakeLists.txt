# orc-rt-connect: what every way of reaching a controller in another process
# shares. That is connection specs and the connector registry, the SimpleRemote
# protocol, and VettedPeer.
#
# Transports are separate libraries built on this one (orc-rt-connect-socket,
# for example), so that which transports an executor can use is decided by what
# it links. Nothing in this library opens a channel of any kind.
set(ORC_RT_CONNECT_SOURCES
  ConnectionSpec.cpp
  ConnectorRegistry.cpp
  sps/SimpleRemoteCA.cpp
  )

add_library(orc-rt-connect-objects OBJECT ${ORC_RT_CONNECT_SOURCES})

set_target_properties(orc-rt-connect-objects PROPERTIES
  POSITION_INDEPENDENT_CODE ${ORC_RT_ENABLE_PIC}
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

target_link_libraries(orc-rt-connect-objects PUBLIC orc-rt-headers)

# Apply RTTI and exceptions compile flags
target_compile_options(orc-rt-connect-objects PRIVATE ${ORC_RT_COMPILE_FLAGS})

# The shipped library. It does not embed Support's objects: it links Bedrock,
# which already carries them, so a process linking both gets one copy rather
# than two.
add_library(orc-rt-connect ${ORC_RT_BEDROCK_LIBRARY_TYPE})

target_link_libraries(orc-rt-connect PUBLIC orc-rt-headers orc-rt-bedrock)

target_link_libraries(orc-rt-connect
  PRIVATE $<BUILD_INTERFACE:orc-rt-connect-objects>)

# TODO: Use common runtimes infrastructure for output and install paths
install(TARGETS orc-rt-connect
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          COMPONENT OrcRT_Development
  PUBLIC_HEADER DESTINATION include COMPONENT OrcRT_Development
)
