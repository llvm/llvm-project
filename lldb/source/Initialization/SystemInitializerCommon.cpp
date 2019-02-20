//===-- SystemInitializerCommon.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Initialization/SystemInitializerCommon.h"

#include "Plugins/DynamicLoader/MacOSX-DYLD/DynamicLoaderMacOSXDYLD.h"
#include "Plugins/DynamicLoader/POSIX-DYLD/DynamicLoaderPOSIXDYLD.h"
#include "Plugins/DynamicLoader/Windows-DYLD/DynamicLoaderWindowsDYLD.h"
#include "Plugins/ExpressionParser/Swift/SwiftREPL.h"
#include "Plugins/Instruction/ARM/EmulateInstructionARM.h"
#include "Plugins/Instruction/MIPS/EmulateInstructionMIPS.h"
#include "Plugins/Instruction/MIPS64/EmulateInstructionMIPS64.h"
#include "Plugins/ObjectContainer/BSD-Archive/ObjectContainerBSDArchive.h"
#include "Plugins/ObjectContainer/Universal-Mach-O/ObjectContainerUniversalMachO.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/SwiftASTContext.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Reproducer.h"
#include "lldb/Utility/Timer.h"

#if defined(__linux__) || defined(__FreeBSD__) || defined(__NetBSD__)
#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"
#endif

#if defined(_MSC_VER)
#include "Plugins/Process/Windows/Common/ProcessWindowsLog.h"
#include "lldb/Host/windows/windows.h"
#endif

#include "llvm/Support/TargetSelect.h"

#include <string>

using namespace lldb_private;
using namespace lldb_private::repro;

SystemInitializerCommon::SystemInitializerCommon() {}

SystemInitializerCommon::~SystemInitializerCommon() {}

llvm::Error
SystemInitializerCommon::Initialize(const InitializerOptions &options) {
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

  // Initialize the reproducer.
  ReproducerMode mode = ReproducerMode::Off;
  if (options.reproducer_capture)
    mode = ReproducerMode::Capture;
  if (options.reproducer_replay)
    mode = ReproducerMode::Replay;

  if (auto e = Reproducer::Initialize(mode, FileSpec(options.reproducer_path)))
    return e;

  // Initialize the file system.
  auto &r = repro::Reproducer::Instance();
  if (repro::Loader *loader = r.GetLoader()) {
    FileSpec vfs_mapping = loader->GetFile<FileInfo>();
    if (vfs_mapping) {
      if (llvm::Error e = FileSystem::Initialize(vfs_mapping))
        return e;
    } else {
      FileSystem::Initialize();
    }
  } else if (repro::Generator *g = r.GetGenerator()) {
    repro::FileProvider &fp = g->GetOrCreate<repro::FileProvider>();
    FileSystem::Initialize(fp.GetFileCollector());
  } else {
    FileSystem::Initialize();
  }

  Log::Initialize();
  HostInfo::Initialize();
  static Timer::Category func_cat(LLVM_PRETTY_FUNCTION);
  Timer scoped_timer(func_cat, LLVM_PRETTY_FUNCTION);

  process_gdb_remote::ProcessGDBRemoteLog::Initialize();

  // Initialize plug-ins
  ClangASTContext::Initialize();
  SwiftASTContext::Initialize();

  SwiftREPL::Initialize();

  ObjectContainerBSDArchive::Initialize();

  EmulateInstructionARM::Initialize();
  EmulateInstructionMIPS::Initialize();
  EmulateInstructionMIPS64::Initialize();

  //----------------------------------------------------------------------
  // Apple/Darwin hosted plugins
  //----------------------------------------------------------------------
  ObjectContainerUniversalMachO::Initialize();

#if defined(__linux__) || defined(__FreeBSD__) || defined(__NetBSD__)
  ProcessPOSIXLog::Initialize();
#endif
#if defined(_MSC_VER)
  ProcessWindowsLog::Initialize();
#endif

  return llvm::Error::success();
}

void SystemInitializerCommon::Terminate() {
  static Timer::Category func_cat(LLVM_PRETTY_FUNCTION);
  Timer scoped_timer(func_cat, LLVM_PRETTY_FUNCTION);
  ObjectContainerBSDArchive::Terminate();

  ClangASTContext::Terminate();
  SwiftASTContext::Terminate();

  SwiftREPL::Terminate();

  EmulateInstructionARM::Terminate();
  EmulateInstructionMIPS::Terminate();
  EmulateInstructionMIPS64::Terminate();

  ObjectContainerUniversalMachO::Terminate();

#if defined(_MSC_VER)
  ProcessWindowsLog::Terminate();
#endif

  HostInfo::Terminate();
  Log::DisableAllLogChannels();
  FileSystem::Terminate();
  Reproducer::Terminate();
}
