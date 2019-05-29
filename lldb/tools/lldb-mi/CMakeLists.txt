if ( CMAKE_SYSTEM_NAME MATCHES "Windows" OR CMAKE_SYSTEM_NAME MATCHES "NetBSD" )
  add_definitions( -DIMPORT_LIBLLDB )
  list(APPEND extra_libs lldbHost)
endif ()

if (HAVE_LIBPTHREAD)
  list(APPEND extra_libs pthread)
endif ()

# We need to include the llvm components we depend on manually, as liblldb does
# not re-export those.
set(LLVM_LINK_COMPONENTS Support)
add_lldb_tool(lldb-mi
  MICmdArgContext.cpp
  MICmdArgSet.cpp
  MICmdArgValBase.cpp
  MICmdArgValConsume.cpp
  MICmdArgValFile.cpp
  MICmdArgValListBase.cpp
  MICmdArgValListOfN.cpp
  MICmdArgValNumber.cpp
  MICmdArgValOptionLong.cpp
  MICmdArgValOptionShort.cpp
  MICmdArgValPrintValues.cpp
  MICmdArgValString.cpp
  MICmdArgValThreadGrp.cpp
  MICmdBase.cpp
  MICmdCommands.cpp
  MICmdCmd.cpp
  MICmdCmdBreak.cpp
  MICmdCmdData.cpp
  MICmdCmdEnviro.cpp
  MICmdCmdExec.cpp
  MICmdCmdFile.cpp
  MICmdCmdGdbInfo.cpp
  MICmdCmdGdbSet.cpp
  MICmdCmdGdbShow.cpp
  MICmdCmdGdbThread.cpp
  MICmdCmdMiscellanous.cpp
  MICmdCmdStack.cpp
  MICmdCmdSupportInfo.cpp
  MICmdCmdSupportList.cpp
  MICmdCmdSymbol.cpp
  MICmdCmdTarget.cpp
  MICmdCmdThread.cpp
  MICmdCmdTrace.cpp
  MICmdCmdVar.cpp
  MICmdData.cpp
  MICmdFactory.cpp
  MICmdInterpreter.cpp
  MICmdInvoker.cpp
  MICmdMgr.cpp
  MICmdMgrSetCmdDeleteCallback.cpp
  MICmnBase.cpp
  MICmnLLDBBroadcaster.cpp
  MICmnLLDBDebugger.cpp
  MICmnLLDBDebuggerHandleEvents.cpp
  MICmnLLDBDebugSessionInfo.cpp
  MICmnLLDBDebugSessionInfoVarObj.cpp
  MICmnLLDBProxySBValue.cpp
  MICmnLLDBUtilSBValue.cpp
  MICmnLog.cpp
  MICmnLogMediumFile.cpp
  MICmnMIOutOfBandRecord.cpp
  MICmnMIResultRecord.cpp
  MICmnMIValue.cpp
  MICmnMIValueConst.cpp
  MICmnMIValueList.cpp
  MICmnMIValueResult.cpp
  MICmnMIValueTuple.cpp
  MICmnResources.cpp
  MICmnStreamStderr.cpp
  MICmnStreamStdin.cpp
  MICmnStreamStdout.cpp
  MICmnThreadMgrStd.cpp
  MIDriver.cpp
  MIDriverBase.cpp
  MIDriverMain.cpp
  MIDriverMgr.cpp
  MIUtilDateTimeStd.cpp
  MIUtilDebug.cpp
  MIUtilFileStd.cpp
  MIUtilMapIdToVariant.cpp
  MIUtilString.cpp
  MIUtilThreadBaseStd.cpp
  MIUtilVariant.cpp

  LINK_LIBS
    liblldb
    ${host_lib}
    ${extra_libs}

  LINK_COMPONENTS
    Support
  )

if(LLDB_BUILD_FRAMEWORK)
  # In the build-tree, we know the exact path to the framework directory.
  # The installed framework can be in different locations.
  get_target_property(framework_build_dir liblldb LIBRARY_OUTPUT_DIRECTORY)
  lldb_setup_rpaths(lldb-mi
    BUILD_RPATH
      "${framework_build_dir}"
    INSTALL_RPATH
      "@loader_path/../../../SharedFrameworks"
      "@loader_path/../../System/Library/PrivateFrameworks"
      "@loader_path/../../Library/PrivateFrameworks"
  )
endif()
