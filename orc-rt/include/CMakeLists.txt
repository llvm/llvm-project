set(ORC_RT_HEADERS
    orc-rt-c/orc-rt.h
    orc-rt/bitmask-enum.h
    orc-rt/error.h
    orc-rt/compiler.h
    orc-rt/math.h
    orc-rt/rtti.h
    orc-rt/span.h
)

# TODO: Switch to filesets when we move to cmake-3.23.
add_library(orc-rt-headers INTERFACE)
target_include_directories(orc-rt-headers INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set_property(TARGET orc-rt-headers
    PROPERTY PUBLIC_HEADER ${ORC_RT_HEADERS}
)
install(TARGETS orc-rt-headers
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/
    COMPONENT OrcRT_Development
)
