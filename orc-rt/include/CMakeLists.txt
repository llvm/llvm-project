set(ORC_RT_HEADERS
    orc-rt-c/CoreTyspe.h
    orc-rt-c/ExternC.h
    orc-rt-c/WrapperFunction.h
    orc-rt-c/orc-rt.h
    orc-rt/BitmaskEnum.h
    orc-rt/Compiler.h
    orc-rt/Error.h
    orc-rt/ExecutorAddress.h
    orc-rt/IntervalMap.h
    orc-rt/IntervalSet.h
    orc-rt/Math.h
    orc-rt/MemoryFlags.h
    orc-rt/RTTI.h
    orc-rt/WrapperFunction.h
    orc-rt/ScopeExit.h
    orc-rt/SimplePackedSerialization.h
    orc-rt/SPSWrapperFunction.h
    orc-rt/bind.h
    orc-rt/bit.h
    orc-rt/move_only_function.h
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
