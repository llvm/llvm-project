//===-- SystemInitializerCommon.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Initialization/SystemInitializerCommon.h"

#include "Plugins/Instruction/ARM/EmulateInstructionARM.h"
#include "Plugins/Instruction/MIPS/EmulateInstructionMIPS.h"
#include "Plugins/Instruction/MIPS64/EmulateInstructionMIPS64.h"
#include "Plugins/ObjectContainer/BSD-Archive/ObjectContainerBSDArchive.h"
#include "Plugins/ObjectContainer/Universal-Mach-O/ObjectContainerUniversalMachO.h"
#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/ObjectFile/PECOFF/ObjectFilePECOFF.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "lldb/Core/Timer.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/Log.h"

#if defined(__APPLE__)
#include "Plugins/ObjectFile/Mach-O/ObjectFileMachO.h"
#endif

#if defined(__linux__) || defined(__FreeBSD__) || defined(__NetBSD__)
#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"
#endif

#if defined(_MSC_VER)
#include "Plugins/Process/Windows/Common/ProcessWindowsLog.h"
#include "lldb/Host/windows/windows.h"
#endif

#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/TargetSelect.h"

#include <string>

using namespace lldb_private;

SystemInitializerCommon::SystemInitializerCommon() {}

SystemInitializerCommon::~SystemInitializerCommon() {}

void SystemInitializerCommon::Initialize() {
#if defined(_MSC_VER)
  const char *disable_crash_dialog_var = getenv("LLDB_DISABLE_CRASH_DIALOG");
  if (disable_crash_dialog_var &&
      llvm::StringRef(disable_crash_dialog_var).equals_lower("true")) {
    // This will prevent Windows from displaying a dialog box requiring user
    // interaction when
    // LLDB crashes.  This is mostly useful when automating LLDB, for example
    // via the test
    // suite, so that a crash in LLDB does not prevent completion of the test
    // suite.
    ::SetErrorMode(GetErrorMode() | SEM_FAILCRITICALERRORS |
                   SEM_NOGPFAULTERRORBOX);

    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
  }
#endif

  llvm::EnablePrettyStackTrace();
  InitializeLog();
  HostInfo::Initialize();
  static Timer::Category func_cat(LLVM_PRETTY_FUNCTION);
  Timer scoped_timer(func_cat, LLVM_PRETTY_FUNCTION);

  process_gdb_remote::ProcessGDBRemoteLog::Initialize();

  // Initialize plug-ins
  ObjectContainerBSDArchive::Initialize();
  ObjectFileELF::Initialize();
  ObjectFilePECOFF::Initialize();

  EmulateInstructionARM::Initialize();
  EmulateInstructionMIPS::Initialize();
  EmulateInstructionMIPS64::Initialize();

  //----------------------------------------------------------------------
  // Apple/Darwin hosted plugins
  //----------------------------------------------------------------------
  ObjectContainerUniversalMachO::Initialize();

#if defined(__APPLE__)
  ObjectFileMachO::Initialize();
#endif
#if defined(__linux__) || defined(__FreeBSD__) || defined(__NetBSD__)
  ProcessPOSIXLog::Initialize();
#endif
#if defined(_MSC_VER)
  ProcessWindowsLog::Initialize();
#endif
}

void SystemInitializerCommon::Terminate() {
  static Timer::Category func_cat(LLVM_PRETTY_FUNCTION);
  Timer scoped_timer(func_cat, LLVM_PRETTY_FUNCTION);
  ObjectContainerBSDArchive::Terminate();
  ObjectFileELF::Terminate();
  ObjectFilePECOFF::Terminate();

  EmulateInstructionARM::Terminate();
  EmulateInstructionMIPS::Terminate();
  EmulateInstructionMIPS64::Terminate();

  ObjectContainerUniversalMachO::Terminate();
#if defined(__APPLE__)
  ObjectFileMachO::Terminate();
#endif

#if defined(_MSC_VER)
  ProcessWindowsLog::Terminate();
#endif

  HostInfo::Terminate();
  Log::DisableAllLogChannels();
}
