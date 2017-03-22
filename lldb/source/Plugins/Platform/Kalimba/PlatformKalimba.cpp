//===-- PlatformKalimba.cpp ---------------------------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PlatformKalimba.h"
#include "lldb/Host/Config.h"

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Error.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb;
using namespace lldb_private;

static uint32_t g_initialize_count = 0;

PlatformSP PlatformKalimba::CreateInstance(bool force, const ArchSpec *arch) {
  bool create = force;
  if (create == false && arch && arch->IsValid()) {
    const llvm::Triple &triple = arch->GetTriple();
    switch (triple.getVendor()) {
    case llvm::Triple::CSR:
      create = true;
      break;

    default:
      break;
    }
  }
  if (create)
    return PlatformSP(new PlatformKalimba(false));
  return PlatformSP();
}

lldb_private::ConstString
PlatformKalimba::GetPluginNameStatic(bool /*is_host*/) {
  static ConstString g_remote_name("kalimba");
  return g_remote_name;
}

const char *PlatformKalimba::GetPluginDescriptionStatic(bool /*is_host*/) {
  return "Kalimba user platform plug-in.";
}

lldb_private::ConstString PlatformKalimba::GetPluginName() {
  return GetPluginNameStatic(false);
}

void PlatformKalimba::Initialize() {
  Platform::Initialize();

  if (g_initialize_count++ == 0) {
    PluginManager::RegisterPlugin(
        PlatformKalimba::GetPluginNameStatic(false),
        PlatformKalimba::GetPluginDescriptionStatic(false),
        PlatformKalimba::CreateInstance);
  }
}

void PlatformKalimba::Terminate() {
  if (g_initialize_count > 0) {
    if (--g_initialize_count == 0) {
      PluginManager::UnregisterPlugin(PlatformKalimba::CreateInstance);
    }
  }

  Platform::Terminate();
}

//------------------------------------------------------------------
/// Default Constructor
//------------------------------------------------------------------
PlatformKalimba::PlatformKalimba(bool is_host)
    : Platform(is_host), // This is the local host platform
      m_remote_platform_sp() {}

//------------------------------------------------------------------
/// Destructor.
///
/// The destructor is virtual since this class is designed to be
/// inherited from by the plug-in instance.
//------------------------------------------------------------------
PlatformKalimba::~PlatformKalimba() {}

bool PlatformKalimba::GetSupportedArchitectureAtIndex(uint32_t idx,
                                                      ArchSpec &arch) {
  if (idx == 0) {
    arch = ArchSpec("kalimba3-csr-unknown");
    return true;
  }
  if (idx == 1) {
    arch = ArchSpec("kalimba4-csr-unknown");
    return true;
  }
  if (idx == 2) {
    arch = ArchSpec("kalimba5-csr-unknown");
    return true;
  }
  return false;
}

void PlatformKalimba::GetStatus(Stream &strm) { Platform::GetStatus(strm); }

size_t
PlatformKalimba::GetSoftwareBreakpointTrapOpcode(Target & /*target*/,
                                                 BreakpointSite * /*bp_site*/) {
  // the target hardware does not support software breakpoints
  return 0;
}

Error PlatformKalimba::LaunchProcess(ProcessLaunchInfo &launch_info) {
  Error error;

  if (IsHost()) {
    error.SetErrorString("native execution is not possible");
  } else {
    error.SetErrorString("the platform is not currently connected");
  }
  return error;
}

lldb::ProcessSP PlatformKalimba::Attach(ProcessAttachInfo &attach_info,
                                        Debugger &debugger, Target *target,
                                        Error &error) {
  lldb::ProcessSP process_sp;
  if (IsHost()) {
    error.SetErrorString("native execution is not possible");
  } else {
    if (m_remote_platform_sp)
      process_sp =
          m_remote_platform_sp->Attach(attach_info, debugger, target, error);
    else
      error.SetErrorString("the platform is not currently connected");
  }
  return process_sp;
}

void PlatformKalimba::CalculateTrapHandlerSymbolNames() {
  // TODO Research this sometime.
}
