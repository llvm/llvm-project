//===-- PlatformRemoteAppleTV.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <string>
#include <vector>

#include "PlatformRemoteAppleTV.h"

#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb;
using namespace lldb_private;

//------------------------------------------------------------------
/// Default Constructor
//------------------------------------------------------------------
PlatformRemoteAppleTV::PlatformRemoteAppleTV()
    : PlatformRemoteDarwinDevice () {}

//------------------------------------------------------------------
// Static Variables
//------------------------------------------------------------------
static uint32_t g_initialize_count = 0;

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
void PlatformRemoteAppleTV::Initialize() {
  PlatformDarwin::Initialize();

  if (g_initialize_count++ == 0) {
    PluginManager::RegisterPlugin(PlatformRemoteAppleTV::GetPluginNameStatic(),
                                  PlatformRemoteAppleTV::GetDescriptionStatic(),
                                  PlatformRemoteAppleTV::CreateInstance);
  }
}

void PlatformRemoteAppleTV::Terminate() {
  if (g_initialize_count > 0) {
    if (--g_initialize_count == 0) {
      PluginManager::UnregisterPlugin(PlatformRemoteAppleTV::CreateInstance);
    }
  }

  PlatformDarwin::Terminate();
}

PlatformSP PlatformRemoteAppleTV::CreateInstance(bool force,
                                                 const ArchSpec *arch) {
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM));
  if (log) {
    const char *arch_name;
    if (arch && arch->GetArchitectureName())
      arch_name = arch->GetArchitectureName();
    else
      arch_name = "<null>";

    const char *triple_cstr =
        arch ? arch->GetTriple().getTriple().c_str() : "<null>";

    log->Printf("PlatformRemoteAppleTV::%s(force=%s, arch={%s,%s})",
                __FUNCTION__, force ? "true" : "false", arch_name, triple_cstr);
  }

  bool create = force;
  if (!create && arch && arch->IsValid()) {
    switch (arch->GetMachine()) {
    case llvm::Triple::arm:
    case llvm::Triple::aarch64:
    case llvm::Triple::thumb: {
      const llvm::Triple &triple = arch->GetTriple();
      llvm::Triple::VendorType vendor = triple.getVendor();
      switch (vendor) {
      case llvm::Triple::Apple:
        create = true;
        break;

#if defined(__APPLE__)
      // Only accept "unknown" for the vendor if the host is Apple and
      // "unknown" wasn't specified (it was just returned because it was NOT
      // specified)
      case llvm::Triple::UnknownVendor:
        create = !arch->TripleVendorWasSpecified();
        break;

#endif
      default:
        break;
      }
      if (create) {
        switch (triple.getOS()) {
        case llvm::Triple::TvOS: // This is the right triple value for Apple TV
                                 // debugging
          break;

        default:
          create = false;
          break;
        }
      }
    } break;
    default:
      break;
    }
  }

  if (create) {
    if (log)
      log->Printf("PlatformRemoteAppleTV::%s() creating platform",
                  __FUNCTION__);

    return lldb::PlatformSP(new PlatformRemoteAppleTV());
  }

  if (log)
    log->Printf("PlatformRemoteAppleTV::%s() aborting creation of platform",
                __FUNCTION__);

  return lldb::PlatformSP();
}

lldb_private::ConstString PlatformRemoteAppleTV::GetPluginNameStatic() {
  static ConstString g_name("remote-tvos");
  return g_name;
}

const char *PlatformRemoteAppleTV::GetDescriptionStatic() {
  return "Remote Apple TV platform plug-in.";
}

bool PlatformRemoteAppleTV::GetSupportedArchitectureAtIndex(uint32_t idx,
                                                            ArchSpec &arch) {
  ArchSpec system_arch(GetSystemArchitecture());

  const ArchSpec::Core system_core = system_arch.GetCore();
  switch (system_core) {
  default:
    switch (idx) {
    case 0:
      arch.SetTriple("arm64-apple-tvos");
      return true;
    case 1:
      arch.SetTriple("armv7s-apple-tvos");
      return true;
    case 2:
      arch.SetTriple("armv7-apple-tvos");
      return true;
    case 3:
      arch.SetTriple("thumbv7s-apple-tvos");
      return true;
    case 4:
      arch.SetTriple("thumbv7-apple-tvos");
      return true;
    default:
      break;
    }
    break;

  case ArchSpec::eCore_arm_arm64:
    switch (idx) {
    case 0:
      arch.SetTriple("arm64-apple-tvos");
      return true;
    case 1:
      arch.SetTriple("armv7s-apple-tvos");
      return true;
    case 2:
      arch.SetTriple("armv7-apple-tvos");
      return true;
    case 3:
      arch.SetTriple("thumbv7s-apple-tvos");
      return true;
    case 4:
      arch.SetTriple("thumbv7-apple-tvos");
      return true;
    default:
      break;
    }
    break;

  case ArchSpec::eCore_arm_armv7s:
    switch (idx) {
    case 0:
      arch.SetTriple("armv7s-apple-tvos");
      return true;
    case 1:
      arch.SetTriple("armv7-apple-tvos");
      return true;
    case 2:
      arch.SetTriple("thumbv7s-apple-tvos");
      return true;
    case 3:
      arch.SetTriple("thumbv7-apple-tvos");
      return true;
    default:
      break;
    }
    break;

  case ArchSpec::eCore_arm_armv7:
    switch (idx) {
    case 0:
      arch.SetTriple("armv7-apple-tvos");
      return true;
    case 1:
      arch.SetTriple("thumbv7-apple-tvos");
      return true;
    default:
      break;
    }
    break;
  }
  arch.Clear();
  return false;
}


void PlatformRemoteAppleTV::GetDeviceSupportDirectoryNames (std::vector<std::string> &dirnames) 
{
    dirnames.clear();
    dirnames.push_back("tvOS DeviceSupport");
}

std::string PlatformRemoteAppleTV::GetPlatformName ()
{
    return "AppleTVOS.platform";
}

