# Headers installed with the runtime.
#
# Headers under orc-rt-internal/ are deliberately not listed: they are not
# installed. Don't add them here to "complete" the list. An installed header
# must not include an internal one, or the installed tree won't compile.
set(ORC_RT_HEADERS
    orc-rt-c/bedrock/Session.h
    orc-rt-c/support/Compiler.h
    orc-rt-c/support/CoreTypes.h
    orc-rt-c/support/Error.h
    orc-rt-c/support/Logging.h
    orc-rt-c/support/WrapperFunction.h
    orc-rt/bedrock/BootstrapInfo.h
    orc-rt/bedrock/ExecutorProcessInfo.h
    orc-rt/bedrock/InProcessControllerAccess.h
    orc-rt/bedrock/NativeDylibManager.h
    orc-rt/bedrock/QueueingRunner.h
    orc-rt/bedrock/Service.h
    orc-rt/bedrock/Session.h
    orc-rt/bedrock/SimpleNativeMemoryMap.h
    orc-rt/bedrock/SimpleRemoteCA.h
    orc-rt/bedrock/SimpleSymbolTable.h
    orc-rt/bedrock/StandaloneMachOUnwindInfoRegistrar.h
    orc-rt/bedrock/TaskGroup.h
    orc-rt/bedrock/ThreadPoolRunner.h
    orc-rt/bedrock/sps/AllSPSCI.h
    orc-rt/bedrock/sps/CallSPSCI.h
    orc-rt/bedrock/sps/GDBJITRegistrarSPSCI.h
    orc-rt/bedrock/sps/MemoryAccessSPSCI.h
    orc-rt/bedrock/sps/NativeDylibManagerSPSCI.h
    orc-rt/bedrock/sps/SimpleNativeMemoryMapSPSCI.h
    orc-rt/bedrock/sps/StandaloneMachOUnwindInfoRegistrarSPSCI.h
    orc-rt/support/AllocAction.h
    orc-rt/support/BitmaskEnum.h
    orc-rt/support/CCallback.h
    orc-rt/support/CallableTraitsHelper.h
    orc-rt/support/Compiler.h
    orc-rt/support/Error.h
    orc-rt/support/ExecutorAddress.h
    orc-rt/support/IntervalMap.h
    orc-rt/support/IntervalSet.h
    orc-rt/support/LockedAccess.h
    orc-rt/support/MacroUtils.h
    orc-rt/support/Math.h
    orc-rt/support/MemoryFlags.h
    orc-rt/support/Proxy.h
    orc-rt/support/RTTI.h
    orc-rt/support/SPSAllocAction.h
    orc-rt/support/SPSMemoryFlags.h
    orc-rt/support/SPSWrapperFunction.h
    orc-rt/support/SPSWrapperFunctionBuffer.h
    orc-rt/support/SimplePackedSerialization.h
    orc-rt/support/StringPool.h
    orc-rt/support/WrapperFunction.h
    orc-rt/support/bind.h
    orc-rt/support/bit.h
    orc-rt/support/iterator_range.h
    orc-rt/support/move_only_function.h
    orc-rt/support/scope_exit.h
    orc-rt/support/span.h
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
