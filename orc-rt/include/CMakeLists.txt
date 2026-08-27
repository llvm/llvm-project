set(ORC_RT_HEADERS
    orc-rt-c/Compiler.h
    orc-rt-c/CoreTyspe.h
    orc-rt-c/Error.h
    orc-rt-c/Logging.h
    orc-rt-c/WrapperFunction.h
    orc-rt-c/orc-rt.h
    orc-rt/bedrock/AllocAction.h
    orc-rt/bedrock/BitmaskEnum.h
    orc-rt/bedrock/BootstrapInfo.h
    orc-rt/bedrock/CCallback.h
    orc-rt/bedrock/Compiler.h
    orc-rt/bedrock/Error.h
    orc-rt/bedrock/ExecutorAddress.h
    orc-rt/bedrock/InProcessControllerAccess.h
    orc-rt/bedrock/IntervalMap.h
    orc-rt/bedrock/IntervalSet.h
    orc-rt/bedrock/LockedAccess.h
    orc-rt/bedrock/MacroUtils.h
    orc-rt/bedrock/Math.h
    orc-rt/bedrock/MemoryFlags.h
    orc-rt/bedrock/NativeDylibManager.h
    orc-rt/bedrock/Proxy.h
    orc-rt/bedrock/QueueingRunner.h
    orc-rt/bedrock/RTTI.h
    orc-rt/bedrock/SPSAllocAction.h
    orc-rt/bedrock/SPSMemoryFlags.h
    orc-rt/bedrock/SPSWrapperFunction.h
    orc-rt/bedrock/SPSWrapperFunctionBuffer.h
    orc-rt/bedrock/Service.h
    orc-rt/bedrock/Session.h
    orc-rt/bedrock/SimpleNativeMemoryMap.h
    orc-rt/bedrock/SimplePackedSerialization.h
    orc-rt/bedrock/SimpleSymbolTable.h
    orc-rt/bedrock/SimpleRemoteCA.h
    orc-rt/bedrock/StandaloneMachOUnwindInfoRegistrar.h
    orc-rt/bedrock/StringPool.h
    orc-rt/bedrock/TaskGroup.h
    orc-rt/bedrock/ThreadPoolRunner.h
    orc-rt/bedrock/WrapperFunction.h
    orc-rt/bedrock/bind.h
    orc-rt/bedrock/bit.h
    orc-rt/bedrock/move_only_function.h
    orc-rt/bedrock/scope_exit.h
    orc-rt/bedrock/span.h
    orc-rt/bedrock/sps-ci/AllSPSCI.h
    orc-rt/bedrock/sps-ci/CallSPSCI.h
    orc-rt/bedrock/sps-ci/GDBJITRegistrarSPSCI.h
    orc-rt/bedrock/sps-ci/MemoryAccessSPSCI.h
    orc-rt/bedrock/sps-ci/NativeDylibManagerSPSCI.h
    orc-rt/bedrock/sps-ci/SimpleNativeMemoryMapSPSCI.h
)

# Add generated config header
set(ORC_RT_GENERATED_HEADERS
    ${CMAKE_CURRENT_BINARY_DIR}/orc-rt-c/config.h
)

# TODO: Switch to filesets when we move to cmake-3.23.
add_library(orc-rt-headers INTERFACE)
target_include_directories(orc-rt-headers INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set_property(TARGET orc-rt-headers
    PROPERTY PUBLIC_HEADER ${ORC_RT_HEADERS} ${ORC_RT_GENERATED_HEADERS}
)
install(TARGETS orc-rt-headers
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/
    COMPONENT OrcRT_Development
)

# Install generated config header
install(FILES ${ORC_RT_GENERATED_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/orc-rt-c/
    COMPONENT OrcRT_Development
)
