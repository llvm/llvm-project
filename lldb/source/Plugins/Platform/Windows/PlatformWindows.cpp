//===-- PlatformWindows.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PlatformWindows.h"

#include <cstdio>
#if defined(_WIN32)
#include "lldb/Host/windows/windows.h"
#include <winsock2.h>
#endif

#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Breakpoint/BreakpointSite.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Expression/UserExpression.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/Status.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(PlatformWindows)

static uint32_t g_initialize_count = 0;

namespace {
class SupportedArchList {
public:
  SupportedArchList() {
    AddArch(ArchSpec("i686-pc-windows"));
    AddArch(HostInfo::GetArchitecture(HostInfo::eArchKindDefault));
    AddArch(HostInfo::GetArchitecture(HostInfo::eArchKind32));
    AddArch(HostInfo::GetArchitecture(HostInfo::eArchKind64));
    AddArch(ArchSpec("i386-pc-windows"));
  }

  size_t Count() const { return m_archs.size(); }

  const ArchSpec &operator[](int idx) { return m_archs[idx]; }

private:
  void AddArch(const ArchSpec &spec) {
    if (llvm::any_of(m_archs, [spec](const ArchSpec &rhs) {
          return spec.IsExactMatch(rhs);
        }))
      return;
    if (spec.IsValid())
      m_archs.push_back(spec);
  }

  std::vector<ArchSpec> m_archs;
};
} // anonymous namespace

PlatformSP PlatformWindows::CreateInstance(bool force,
                                           const lldb_private::ArchSpec *arch) {
  // The only time we create an instance is when we are creating a remote
  // windows platform
  const bool is_host = false;

  bool create = force;
  if (!create && arch && arch->IsValid()) {
    const llvm::Triple &triple = arch->GetTriple();
    switch (triple.getVendor()) {
    case llvm::Triple::PC:
      create = true;
      break;

    case llvm::Triple::UnknownVendor:
      create = !arch->TripleVendorWasSpecified();
      break;

    default:
      break;
    }

    if (create) {
      switch (triple.getOS()) {
      case llvm::Triple::Win32:
        break;

      case llvm::Triple::UnknownOS:
        create = arch->TripleOSWasSpecified();
        break;

      default:
        create = false;
        break;
      }
    }
  }
  if (create)
    return PlatformSP(new PlatformWindows(is_host));
  return PlatformSP();
}

lldb_private::ConstString PlatformWindows::GetPluginNameStatic(bool is_host) {
  if (is_host) {
    static ConstString g_host_name(Platform::GetHostPlatformName());
    return g_host_name;
  } else {
    static ConstString g_remote_name("remote-windows");
    return g_remote_name;
  }
}

const char *PlatformWindows::GetPluginDescriptionStatic(bool is_host) {
  return is_host ? "Local Windows user platform plug-in."
                 : "Remote Windows user platform plug-in.";
}

void PlatformWindows::Initialize() {
  Platform::Initialize();

  if (g_initialize_count++ == 0) {
#if defined(_WIN32)
    // Force a host flag to true for the default platform object.
    PlatformSP default_platform_sp(new PlatformWindows(true));
    default_platform_sp->SetSystemArchitecture(HostInfo::GetArchitecture());
    Platform::SetHostPlatform(default_platform_sp);
#endif
    PluginManager::RegisterPlugin(
        PlatformWindows::GetPluginNameStatic(false),
        PlatformWindows::GetPluginDescriptionStatic(false),
        PlatformWindows::CreateInstance);
  }
}

void PlatformWindows::Terminate() {
  if (g_initialize_count > 0) {
    if (--g_initialize_count == 0) {
      PluginManager::UnregisterPlugin(PlatformWindows::CreateInstance);
    }
  }

  Platform::Terminate();
}

/// Default Constructor
PlatformWindows::PlatformWindows(bool is_host) : RemoteAwarePlatform(is_host) {}

Status PlatformWindows::ConnectRemote(Args &args) {
  Status error;
  if (IsHost()) {
    error.SetErrorStringWithFormatv(
        "can't connect to the host platform '{0}', always connected",
        GetPluginName());
  } else {
    if (!m_remote_platform_sp)
      m_remote_platform_sp =
          Platform::Create(ConstString("remote-gdb-server"), error);

    if (m_remote_platform_sp) {
      if (error.Success()) {
        if (m_remote_platform_sp) {
          error = m_remote_platform_sp->ConnectRemote(args);
        } else {
          error.SetErrorString(
              "\"platform connect\" takes a single argument: <connect-url>");
        }
      }
    } else
      error.SetErrorString("failed to create a 'remote-gdb-server' platform");

    if (error.Fail())
      m_remote_platform_sp.reset();
  }

  return error;
}

Status PlatformWindows::DisconnectRemote() {
  Status error;

  if (IsHost()) {
    error.SetErrorStringWithFormatv(
        "can't disconnect from the host platform '{0}', always connected",
        GetPluginName());
  } else {
    if (m_remote_platform_sp)
      error = m_remote_platform_sp->DisconnectRemote();
    else
      error.SetErrorString("the platform is not currently connected");
  }
  return error;
}

Status PlatformWindows::EvaluateLoaderExpression(Process *process,
                                                 const char *expression,
                                                 ValueObjectSP &value) {
  // FIXME(compnerd) `-fdeclspec` is not passed to the clang instance?
  static const char kLoaderDecls[] =
      R"(
          // libloaderapi.h

          // WINBASEAPI BOOL WINAPI FreeModule(HMODULE);
          extern "C" /* __declspec(dllimport) */ BOOL __stdcall FreeModule(void *hLibModule);

          // WINBASEAPI HMODULE WINAPI LoadLibraryA(LPCSTR);
          extern "C" /* __declspec(dllimport) */ void * __stdcall LoadLibraryA(const char *);
        )";

  if (DynamicLoader *loader = process->GetDynamicLoader()) {
    Status result = loader->CanLoadImage();
    if (result.Fail())
      return result;
  }

  ThreadSP thread = process->GetThreadList().GetExpressionExecutionThread();
  if (!thread)
    return Status("selected thread is invalid");

  StackFrameSP frame = thread->GetStackFrameAtIndex(0);
  if (!frame)
    return Status("frame 0 is invalid");

  ExecutionContext context;
  frame->CalculateExecutionContext(context);

  EvaluateExpressionOptions options;
  options.SetUnwindOnError(true);
  options.SetIgnoreBreakpoints(true);
  options.SetExecutionPolicy(eExecutionPolicyAlways);
  options.SetLanguage(eLanguageTypeC_plus_plus);
  // LoadLibrary{A,W}/FreeLibrary cannot raise exceptions which we can handle.
  // They may potentially throw SEH exceptions which we do not know how to
  // handle currently.
  options.SetTrapExceptions(false);
  options.SetTimeout(process->GetUtilityExpressionTimeout());

  Status error;
  ExpressionResults result = UserExpression::Evaluate(
      context, options, expression, kLoaderDecls, value, error);
  if (result != eExpressionCompleted)
    return error;

  if (value->GetError().Fail())
    return value->GetError();

  return Status();
}

uint32_t PlatformWindows::DoLoadImage(Process *process,
                                      const FileSpec &remote_file,
                                      const std::vector<std::string> *paths,
                                      Status &error, FileSpec *loaded_path) {
  if (loaded_path)
    loaded_path->Clear();

  StreamString expression;
  expression.Printf("LoadLibraryA(\"%s\")", remote_file.GetPath().c_str());

  ValueObjectSP value;
  Status result =
      EvaluateLoaderExpression(process, expression.GetData(), value);
  if (result.Fail())
    return LLDB_INVALID_IMAGE_TOKEN;

  Scalar scalar;
  if (value->ResolveValue(scalar)) {
    lldb::addr_t address = scalar.ULongLong();
    if (address == 0)
      return LLDB_INVALID_IMAGE_TOKEN;
    return process->AddImageToken(address);
  }
  return LLDB_INVALID_IMAGE_TOKEN;
}

Status PlatformWindows::UnloadImage(Process *process, uint32_t image_token) {
  const addr_t address = process->GetImagePtrFromToken(image_token);
  if (address == LLDB_INVALID_ADDRESS)
    return Status("invalid image token");

  StreamString expression;
  expression.Printf("FreeLibrary((HMODULE)0x%" PRIx64 ")", address);

  ValueObjectSP value;
  Status result =
      EvaluateLoaderExpression(process, expression.GetData(), value);
  if (result.Fail())
    return result;

  Scalar scalar;
  if (value->ResolveValue(scalar)) {
    if (scalar.UInt(1))
      return Status("expression failed: \"%s\"", expression.GetData());
    process->ResetImageToken(image_token);
  }

  return Status();
}

ProcessSP PlatformWindows::DebugProcess(ProcessLaunchInfo &launch_info,
                                        Debugger &debugger, Target &target,
                                        Status &error) {
  // Windows has special considerations that must be followed when launching or
  // attaching to a process.  The key requirement is that when launching or
  // attaching to a process, you must do it from the same the thread that will
  // go into a permanent loop which will then receive debug events from the
  // process.  In particular, this means we can't use any of LLDB's generic
  // mechanisms to do it for us, because it doesn't have the special knowledge
  // required for setting up the background thread or passing the right flags.
  //
  // Another problem is that that LLDB's standard model for debugging a process
  // is to first launch it, have it stop at the entry point, and then attach to
  // it.  In Windows this doesn't quite work, you have to specify as an
  // argument to CreateProcess() that you're going to debug the process.  So we
  // override DebugProcess here to handle this.  Launch operations go directly
  // to the process plugin, and attach operations almost go directly to the
  // process plugin (but we hijack the events first).  In essence, we
  // encapsulate all the logic of Launching and Attaching in the process
  // plugin, and PlatformWindows::DebugProcess is just a pass-through to get to
  // the process plugin.

  if (IsRemote()) {
    if (m_remote_platform_sp)
      return m_remote_platform_sp->DebugProcess(launch_info, debugger, target,
                                                error);
    else
      error.SetErrorString("the platform is not currently connected");
  }

  if (launch_info.GetProcessID() != LLDB_INVALID_PROCESS_ID) {
    // This is a process attach.  Don't need to launch anything.
    ProcessAttachInfo attach_info(launch_info);
    return Attach(attach_info, debugger, &target, error);
  } else {
    ProcessSP process_sp = target.CreateProcess(
        launch_info.GetListener(), launch_info.GetProcessPluginName(), nullptr,
        false);

    // We need to launch and attach to the process.
    launch_info.GetFlags().Set(eLaunchFlagDebug);
    if (process_sp)
      error = process_sp->Launch(launch_info);

    return process_sp;
  }
}

lldb::ProcessSP PlatformWindows::Attach(ProcessAttachInfo &attach_info,
                                        Debugger &debugger, Target *target,
                                        Status &error) {
  error.Clear();
  lldb::ProcessSP process_sp;
  if (!IsHost()) {
    if (m_remote_platform_sp)
      process_sp =
          m_remote_platform_sp->Attach(attach_info, debugger, target, error);
    else
      error.SetErrorString("the platform is not currently connected");
    return process_sp;
  }

  if (target == nullptr) {
    TargetSP new_target_sp;
    FileSpec emptyFileSpec;
    ArchSpec emptyArchSpec;

    error = debugger.GetTargetList().CreateTarget(
        debugger, "", "", eLoadDependentsNo, nullptr, new_target_sp);
    target = new_target_sp.get();
  }

  if (!target || error.Fail())
    return process_sp;

  const char *plugin_name = attach_info.GetProcessPluginName();
  process_sp = target->CreateProcess(
      attach_info.GetListenerForProcess(debugger), plugin_name, nullptr, false);

  process_sp->HijackProcessEvents(attach_info.GetHijackListener());
  if (process_sp)
    error = process_sp->Attach(attach_info);

  return process_sp;
}

bool PlatformWindows::GetSupportedArchitectureAtIndex(uint32_t idx,
                                                      ArchSpec &arch) {
  static SupportedArchList architectures;

  if (idx >= architectures.Count())
    return false;
  arch = architectures[idx];
  return true;
}

void PlatformWindows::GetStatus(Stream &strm) {
  Platform::GetStatus(strm);

#ifdef _WIN32
  llvm::VersionTuple version = HostInfo::GetOSVersion();
  strm << "      Host: Windows " << version.getAsString() << '\n';
#endif
}

bool PlatformWindows::CanDebugProcess() { return true; }

ConstString PlatformWindows::GetFullNameForDylib(ConstString basename) {
  if (basename.IsEmpty())
    return basename;

  StreamString stream;
  stream.Printf("%s.dll", basename.GetCString());
  return ConstString(stream.GetString());
}

size_t
PlatformWindows::GetSoftwareBreakpointTrapOpcode(Target &target,
                                                 BreakpointSite *bp_site) {
  ArchSpec arch = target.GetArchitecture();
  assert(arch.IsValid());
  const uint8_t *trap_opcode = nullptr;
  size_t trap_opcode_size = 0;

  switch (arch.GetMachine()) {
  case llvm::Triple::aarch64: {
    static const uint8_t g_aarch64_opcode[] = {0x00, 0x00, 0x3e, 0xd4}; // brk #0xf000
    trap_opcode = g_aarch64_opcode;
    trap_opcode_size = sizeof(g_aarch64_opcode);

    if (bp_site->SetTrapOpcode(trap_opcode, trap_opcode_size))
      return trap_opcode_size;
    return 0;
  } break;

  case llvm::Triple::arm:
  case llvm::Triple::thumb: {
    static const uint8_t g_thumb_opcode[] = {0xfe, 0xde}; // udf #0xfe
    trap_opcode = g_thumb_opcode;
    trap_opcode_size = sizeof(g_thumb_opcode);

    if (bp_site->SetTrapOpcode(trap_opcode, trap_opcode_size))
      return trap_opcode_size;
    return 0;
  } break;

  default:
    return Platform::GetSoftwareBreakpointTrapOpcode(target, bp_site);
  }
}
