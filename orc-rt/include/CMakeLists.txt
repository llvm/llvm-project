set(ORC_RT_HEADERS
    orc-rt-c/Compiler.h
    orc-rt-c/CoreTypes.h
    orc-rt-c/Error.h
    orc-rt-c/Logging.h
    orc-rt-c/Session.h
    orc-rt-c/WrapperFunction.h
    orc-rt-utils/CommandLine.h
    orc-rt/AllocAction.h
    orc-rt/BitmaskEnum.h
    orc-rt/BootstrapInfo.h
    orc-rt/CCallback.h
    orc-rt/CallableTraitsHelper.h
    orc-rt/Compiler.h
    orc-rt/Endian.h
    orc-rt/Error.h
    orc-rt/ExecutorAddress.h
    orc-rt/ExecutorProcessInfo.h
    orc-rt/InProcessControllerAccess.h
    orc-rt/IntervalMap.h
    orc-rt/IntervalSet.h
    orc-rt/LockedAccess.h
    orc-rt/MacroUtils.h
    orc-rt/Math.h
    orc-rt/MemoryFlags.h
    orc-rt/NativeDylibManager.h
    orc-rt/QueueingRunner.h
    orc-rt/RTTI.h
    orc-rt/SPSAllocAction.h
    orc-rt/SPSMemoryFlags.h
    orc-rt/SPSWrapperFunction.h
    orc-rt/SPSWrapperFunctionBuffer.h
    orc-rt/Service.h
    orc-rt/Session.h
    orc-rt/SimpleNativeMemoryMap.h
    orc-rt/SimplePackedSerialization.h
    orc-rt/SimpleSymbolTable.h
    orc-rt/StandaloneMachOUnwindInfoRegistrar.h
    orc-rt/StringExtras.h
    orc-rt/TaskGroup.h
    orc-rt/ThreadPoolRunner.h
    orc-rt/WrapperFunction.h
    orc-rt/bind.h
    orc-rt/bit.h
    orc-rt/iterator_range.h
    orc-rt/move_only_function.h
    orc-rt/scope_exit.h
    orc-rt/span.h
    orc-rt/sps-ci/AllSPSCI.h
    orc-rt/sps-ci/CallSPSCI.h
    orc-rt/sps-ci/GDBJITRegistrarSPSCI.h
    orc-rt/sps-ci/MemoryAccessSPSCI.h
    orc-rt/sps-ci/NativeDylibManagerSPSCI.h
    orc-rt/sps-ci/SimpleNativeMemoryMapSPSCI.h
    orc-rt/sps-ci/StandaloneMachOUnwindInfoRegistrarSPSCI.h
)
#LLVM prefers not to use globs, so we just use it to assert that the list is up to date.
file(GLOB_RECURSE ORC_RT_DISK_INCLUDES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

set(ORC_RT_NOT_INCLUDED ${ORC_RT_DISK_INCLUDES})
if(ORC_RT_HEADERS)
  list(REMOVE_ITEM ORC_RT_NOT_INCLUDED ${ORC_RT_HEADERS})
endif()

set(ORC_RT_STALE ${ORC_RT_HEADERS})
if(ORC_RT_DISK_INCLUDES)
  list(REMOVE_ITEM ORC_RT_STALE ${ORC_RT_DISK_INCLUDES})
endif()

if(ORC_RT_NOT_INCLUDED OR ORC_RT_STALE)
string(JOIN "\n  " _msg_unlisted ${ORC_RT_NOT_INCLUDED})
string(JOIN "\n  " _msg_stale ${ORC_RT_STALE})
  message(FATAL_ERROR
    "ORC_RT_HEADERS is out of sync with disk/.\n"
    "On disk but not listed:\n  ${_msg_unlisted}\n"
    "Listed but not on disk:\n  ${_msg_stale}")
endif()

# Add generated config header
set(ORC_RT_GENERATED_HEADERS
    ${CMAKE_CURRENT_BINARY_DIR}/orc-rt-c/config.h
)

add_library(orc-rt-headers INTERFACE)
target_sources(orc-rt-headers INTERFACE
  FILE_SET HEADERS
  BASE_DIRS
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
  FILES
    ${ORC_RT_HEADERS}
    ${ORC_RT_GENERATED_HEADERS}
)

install(TARGETS orc-rt-headers
  FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT OrcRT_Development
)
