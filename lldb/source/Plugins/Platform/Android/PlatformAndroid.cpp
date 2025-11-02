//===-- PlatformAndroid.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/UriParser.h"
#include "lldb/ValueObject/ValueObject.h"

#include "AdbClient.h"
#include "PlatformAndroid.h"
#include "PlatformAndroidRemoteGDBServer.h"
#include "lldb/Target/Target.h"
#include <optional>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_android;
using namespace std::chrono;

LLDB_PLUGIN_DEFINE(PlatformAndroid)

namespace {

#define LLDB_PROPERTIES_android
#include "PlatformAndroidProperties.inc"

enum {
#define LLDB_PROPERTIES_android
#include "PlatformAndroidPropertiesEnum.inc"
};

class PluginProperties : public Properties {
public:
  PluginProperties() {
    m_collection_sp = std::make_shared<OptionValueProperties>(
        PlatformAndroid::GetPluginNameStatic(false));
    m_collection_sp->Initialize(g_android_properties);
  }
};

static PluginProperties &GetGlobalProperties() {
  static PluginProperties g_settings;
  return g_settings;
}

uint32_t g_initialize_count = 0;
const unsigned int g_android_default_cache_size =
    2048; // Fits inside 4k adb packet.

} // end of anonymous namespace

void PlatformAndroid::Initialize() {
  PlatformLinux::Initialize();

  if (g_initialize_count++ == 0) {
#if defined(__ANDROID__)
    PlatformSP default_platform_sp(new PlatformAndroid(true));
    default_platform_sp->SetSystemArchitecture(HostInfo::GetArchitecture());
    Platform::SetHostPlatform(default_platform_sp);
#endif
    PluginManager::RegisterPlugin(
        PlatformAndroid::GetPluginNameStatic(false),
        PlatformAndroid::GetPluginDescriptionStatic(false),
        PlatformAndroid::CreateInstance, PlatformAndroid::DebuggerInitialize);
  }
}

void PlatformAndroid::Terminate() {
  if (g_initialize_count > 0) {
    if (--g_initialize_count == 0) {
      PluginManager::UnregisterPlugin(PlatformAndroid::CreateInstance);
    }
  }

  PlatformLinux::Terminate();
}

PlatformSP PlatformAndroid::CreateInstance(bool force, const ArchSpec *arch) {
  Log *log = GetLog(LLDBLog::Platform);
  if (log) {
    const char *arch_name;
    if (arch && arch->GetArchitectureName())
      arch_name = arch->GetArchitectureName();
    else
      arch_name = "<null>";

    const char *triple_cstr =
        arch ? arch->GetTriple().getTriple().c_str() : "<null>";

    LLDB_LOGF(log, "PlatformAndroid::%s(force=%s, arch={%s,%s})", __FUNCTION__,
              force ? "true" : "false", arch_name, triple_cstr);
  }

  bool create = force;
  if (!create && arch && arch->IsValid()) {
    const llvm::Triple &triple = arch->GetTriple();
    switch (triple.getVendor()) {
    case llvm::Triple::PC:
      create = true;
      break;

#if defined(__ANDROID__)
    // Only accept "unknown" for the vendor if the host is android and if
    // "unknown" wasn't specified (it was just returned because it was NOT
    // specified).
    case llvm::Triple::VendorType::UnknownVendor:
      create = !arch->TripleVendorWasSpecified();
      break;
#endif
    default:
      break;
    }

    if (create) {
      switch (triple.getEnvironment()) {
      case llvm::Triple::Android:
        break;

#if defined(__ANDROID__)
      // Only accept "unknown" for the OS if the host is android and it
      // "unknown" wasn't specified (it was just returned because it was NOT
      // specified)
      case llvm::Triple::EnvironmentType::UnknownEnvironment:
        create = !arch->TripleEnvironmentWasSpecified();
        break;
#endif
      default:
        create = false;
        break;
      }
    }
  }

  if (create) {
    LLDB_LOGF(log, "PlatformAndroid::%s() creating remote-android platform",
              __FUNCTION__);
    return PlatformSP(new PlatformAndroid(false));
  }

  LLDB_LOGF(
      log, "PlatformAndroid::%s() aborting creation of remote-android platform",
      __FUNCTION__);

  return PlatformSP();
}

void PlatformAndroid::DebuggerInitialize(Debugger &debugger) {
  if (!PluginManager::GetSettingForPlatformPlugin(debugger,
                                                  GetPluginNameStatic(false))) {
    PluginManager::CreateSettingForPlatformPlugin(
        debugger, GetGlobalProperties().GetValueProperties(),
        "Properties for the Android platform plugin.",
        /*is_global_property=*/true);
  }
}

PlatformAndroid::PlatformAndroid(bool is_host)
    : PlatformLinux(is_host), m_sdk_version(0) {}

llvm::StringRef PlatformAndroid::GetPluginDescriptionStatic(bool is_host) {
  if (is_host)
    return "Local Android user platform plug-in.";
  return "Remote Android user platform plug-in.";
}

Status PlatformAndroid::ConnectRemote(Args &args) {
  m_device_id.clear();

  if (IsHost())
    return Status::FromErrorString(
        "can't connect to the host platform, always connected");

  if (!m_remote_platform_sp)
    m_remote_platform_sp = PlatformSP(new PlatformAndroidRemoteGDBServer());

  const char *url = args.GetArgumentAtIndex(0);
  if (!url)
    return Status::FromErrorString("URL is null.");
  std::optional<URI> parsed_url = URI::Parse(url);
  if (!parsed_url)
    return Status::FromErrorStringWithFormat("Invalid URL: %s", url);
  if (parsed_url->hostname != "localhost")
    m_device_id = parsed_url->hostname.str();

  auto error = PlatformLinux::ConnectRemote(args);
  if (error.Success()) {
    auto resolved_device_id_or_error = AdbClient::ResolveDeviceID(m_device_id);
    if (!resolved_device_id_or_error)
      return Status::FromError(resolved_device_id_or_error.takeError());
    m_device_id = *resolved_device_id_or_error;
  }
  return error;
}

Status PlatformAndroid::GetFile(const FileSpec &source,
                                const FileSpec &destination) {
  if (IsHost() || !m_remote_platform_sp)
    return PlatformLinux::GetFile(source, destination);

  FileSpec source_spec(source.GetPath(false), FileSpec::Style::posix);
  if (source_spec.IsRelative())
    source_spec = GetRemoteWorkingDirectory().CopyByAppendingPathComponent(
        source_spec.GetPathAsConstString(false).GetStringRef());

  Status error;
  auto sync_service = GetSyncService(error);

  // If sync service is available, try to use it
  if (error.Success() && sync_service) {
    uint32_t mode = 0, size = 0, mtime = 0;
    error = sync_service->Stat(source_spec, mode, size, mtime);
    if (error.Success()) {
      if (mode != 0)
        return sync_service->PullFile(source_spec, destination);

      // mode == 0 can signify that adbd cannot access the file due security
      // constraints - fall through to try "cat ..." as a fallback.
      Log *log = GetLog(LLDBLog::Platform);
      LLDB_LOGF(log, "Got mode == 0 on '%s': try to get file via 'shell cat'",
                source_spec.GetPath(false).c_str());
    }
  }

  // Fallback to shell cat command if sync service failed or returned mode == 0
  std::string source_file = source_spec.GetPath(false);

  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOGF(log, "Using shell cat fallback for '%s'", source_file.c_str());

  if (strchr(source_file.c_str(), '\'') != nullptr)
    return Status::FromErrorString(
        "Doesn't support single-quotes in filenames");

  AdbClientUP adb(GetAdbClient(error));
  if (error.Fail())
    return error;

  char cmd[PATH_MAX];
  snprintf(cmd, sizeof(cmd), "%scat '%s'", GetRunAs().c_str(),
           source_file.c_str());

  return adb->ShellToFile(cmd, minutes(1), destination);
}

Status PlatformAndroid::PutFile(const FileSpec &source,
                                const FileSpec &destination, uint32_t uid,
                                uint32_t gid) {
  if (IsHost() || !m_remote_platform_sp)
    return PlatformLinux::PutFile(source, destination, uid, gid);

  FileSpec destination_spec(destination.GetPath(false), FileSpec::Style::posix);
  if (destination_spec.IsRelative())
    destination_spec = GetRemoteWorkingDirectory().CopyByAppendingPathComponent(
        destination_spec.GetPath(false));

  // TODO: Set correct uid and gid on remote file.
  Status error;
  auto sync_service = GetSyncService(error);
  if (error.Fail())
    return error;
  return sync_service->PushFile(source, destination_spec);
}

const char *PlatformAndroid::GetCacheHostname() { return m_device_id.c_str(); }

Status PlatformAndroid::DownloadModuleSlice(const FileSpec &src_file_spec,
                                            const uint64_t src_offset,
                                            const uint64_t src_size,
                                            const FileSpec &dst_file_spec) {
  std::string source_file = src_file_spec.GetPath(false);
  if (source_file.empty())
    return Status::FromErrorString("Source file path cannot be empty");

  std::string destination_file = dst_file_spec.GetPath(false);
  if (destination_file.empty())
    return Status::FromErrorString("Destination file path cannot be empty");

  // In Android API level 23 and above, dynamic loader is able to load .so
  // file directly from APK. In that case, src_offset will be an non-zero.
  if (src_offset == 0) // Use GetFile for a normal file.
    return GetFile(src_file_spec, dst_file_spec);

  if (source_file.find('\'') != std::string::npos)
    return Status::FromErrorString(
        "Doesn't support single-quotes in filenames");

  // For zip .so file, src_file_spec will be "zip_path!/so_path".
  // Extract "zip_path" from the source_file.
  static constexpr llvm::StringLiteral k_zip_separator("!/");
  size_t pos = source_file.find(k_zip_separator);
  if (pos != std::string::npos)
    source_file.resize(pos);

  Status error;
  AdbClientUP adb(GetAdbClient(error));
  if (error.Fail())
    return error;

  // Use 'shell dd' to download the file slice with the offset and size.
  char cmd[PATH_MAX];
  snprintf(cmd, sizeof(cmd),
           "%sdd if='%s' iflag=skip_bytes,count_bytes "
           "skip=%" PRIu64 " count=%" PRIu64 " status=none",
           GetRunAs().c_str(), source_file.c_str(), src_offset, src_size);

  return adb->ShellToFile(cmd, minutes(1), dst_file_spec);
}

Status PlatformAndroid::DisconnectRemote() {
  Status error = PlatformLinux::DisconnectRemote();
  if (error.Success()) {
    m_device_id.clear();
    m_sdk_version = 0;
  }
  return error;
}

uint32_t PlatformAndroid::GetDefaultMemoryCacheLineSize() {
  return g_android_default_cache_size;
}

uint32_t PlatformAndroid::GetSdkVersion() {
  if (!IsConnected())
    return 0;

  if (m_sdk_version != 0)
    return m_sdk_version;

  std::string version_string;
  Status error;
  AdbClientUP adb(GetAdbClient(error));
  if (error.Fail())
    return 0;
  error =
      adb->Shell("getprop ro.build.version.sdk", seconds(5), &version_string);
  version_string = llvm::StringRef(version_string).trim().str();

  if (error.Fail() || version_string.empty()) {
    Log *log = GetLog(LLDBLog::Platform);
    LLDB_LOGF(log, "Get SDK version failed. (error: %s, output: %s)",
              error.AsCString(), version_string.c_str());
    return 0;
  }

  // FIXME: improve error handling
  llvm::to_integer(version_string, m_sdk_version);
  return m_sdk_version;
}

Status PlatformAndroid::DownloadSymbolFile(const lldb::ModuleSP &module_sp,
                                           const FileSpec &dst_file_spec) {
  // For oat file we can try to fetch additional debug info from the device
  llvm::StringRef extension = module_sp->GetFileSpec().GetFileNameExtension();
  if (extension != ".oat" && extension != ".odex")
    return Status::FromErrorString(
        "Symbol file downloading only supported for oat and odex files");

  // If we have no information about the platform file we can't execute oatdump
  if (!module_sp->GetPlatformFileSpec())
    return Status::FromErrorString("No platform file specified");

  // Symbolizer isn't available before SDK version 23
  if (GetSdkVersion() < 23)
    return Status::FromErrorString(
        "Symbol file generation only supported on SDK 23+");

  // If we already have symtab then we don't have to try and generate one
  if (module_sp->GetSectionList()->FindSectionByName(ConstString(".symtab")) !=
      nullptr)
    return Status::FromErrorString("Symtab already available in the module");

  Status error;
  AdbClientUP adb(GetAdbClient(error));
  if (error.Fail())
    return error;
  std::string tmpdir;
  error = adb->Shell("mktemp --directory --tmpdir /data/local/tmp", seconds(5),
                     &tmpdir);
  if (error.Fail() || tmpdir.empty())
    return Status::FromErrorStringWithFormat(
        "Failed to generate temporary directory on the device (%s)",
        error.AsCString());
  tmpdir = llvm::StringRef(tmpdir).trim().str();

  // Create file remover for the temporary directory created on the device
  std::unique_ptr<std::string, std::function<void(std::string *)>>
      tmpdir_remover(&tmpdir, [&adb](std::string *s) {
        StreamString command;
        command.Printf("rm -rf %s", s->c_str());
        Status error = adb->Shell(command.GetData(), seconds(5), nullptr);

        Log *log = GetLog(LLDBLog::Platform);
        if (log && error.Fail())
          LLDB_LOGF(log, "Failed to remove temp directory: %s",
                    error.AsCString());
      });

  FileSpec symfile_platform_filespec(tmpdir);
  symfile_platform_filespec.AppendPathComponent("symbolized.oat");

  // Execute oatdump on the remote device to generate a file with symtab
  StreamString command;
  command.Printf("oatdump --symbolize=%s --output=%s",
                 module_sp->GetPlatformFileSpec().GetPath(false).c_str(),
                 symfile_platform_filespec.GetPath(false).c_str());
  error = adb->Shell(command.GetData(), minutes(1), nullptr);
  if (error.Fail())
    return Status::FromErrorStringWithFormat("Oatdump failed: %s",
                                             error.AsCString());

  // Download the symbolfile from the remote device
  return GetFile(symfile_platform_filespec, dst_file_spec);
}

bool PlatformAndroid::GetRemoteOSVersion() {
  m_os_version = llvm::VersionTuple(GetSdkVersion());
  return !m_os_version.empty();
}

llvm::StringRef
PlatformAndroid::GetLibdlFunctionDeclarations(lldb_private::Process *process) {
  SymbolContextList matching_symbols;
  std::vector<const char *> dl_open_names = {"__dl_dlopen", "dlopen"};
  const char *dl_open_name = nullptr;
  Target &target = process->GetTarget();
  for (auto *name : dl_open_names) {
    target.GetImages().FindFunctionSymbols(
        ConstString(name), eFunctionNameTypeFull, matching_symbols);
    if (matching_symbols.GetSize()) {
      dl_open_name = name;
      break;
    }
  }
  // Older platform versions have the dl function symbols mangled
  if (dl_open_name == dl_open_names[0])
    return R"(
              extern "C" void* dlopen(const char*, int) asm("__dl_dlopen");
              extern "C" void* dlsym(void*, const char*) asm("__dl_dlsym");
              extern "C" int   dlclose(void*) asm("__dl_dlclose");
              extern "C" char* dlerror(void) asm("__dl_dlerror");
             )";

  return PlatformPOSIX::GetLibdlFunctionDeclarations(process);
}

PlatformAndroid::AdbClientUP PlatformAndroid::GetAdbClient(Status &error) {
  AdbClientUP adb = std::make_unique<AdbClient>(m_device_id);
  error = adb->Connect();
  return adb;
}

llvm::StringRef PlatformAndroid::GetPropertyPackageName() {
  return GetGlobalProperties().GetPropertyAtIndexAs<llvm::StringRef>(
      ePropertyPlatformPackageName, "");
}

std::string PlatformAndroid::GetRunAs() {
  llvm::StringRef run_as = GetPropertyPackageName();
  if (!run_as.empty()) {
    // When LLDB fails to pull file from a package directory due to security
    // constraint, user needs to set the package name to
    // 'platform.plugin.remote-android.package-name' property in order to run
    // shell commands as the package user using 'run-as' (e.g. to get file with
    // 'cat' and 'dd').
    // https://cs.android.com/android/platform/superproject/+/master:
    // system/core/run-as/run-as.cpp;l=39-61;
    // drc=4a77a84a55522a3b122f9c63ef0d0b8a6a131627
    return std::string("run-as '") + run_as.str() + "' ";
  }
  return run_as.str();
}

// Helper function to populate process status information from
// /proc/[pid]/status
void PlatformAndroid::PopulateProcessStatusInfo(
    lldb::pid_t pid, ProcessInstanceInfo &process_info) {
  // Read /proc/[pid]/status to get parent PID, UIDs, and GIDs
  Status error;
  AdbClientUP status_adb = GetAdbClient(error);
  if (error.Fail())
    return;

  std::string status_output;
  StreamString status_cmd;
  status_cmd.Printf(
      "cat /proc/%llu/status 2>/dev/null | grep -E '^(PPid|Uid|Gid):'",
      static_cast<unsigned long long>(pid));
  Status status_error =
      status_adb->Shell(status_cmd.GetData(), seconds(5), &status_output);

  if (status_error.Fail() || status_output.empty())
    return;

  llvm::SmallVector<llvm::StringRef, 16> lines;
  llvm::StringRef(status_output).split(lines, '\n');

  for (llvm::StringRef line : lines) {
    line = line.trim();
    if (line.starts_with("PPid:")) {
      llvm::StringRef ppid_str = line.substr(5).trim();
      lldb::pid_t ppid;
      if (llvm::to_integer(ppid_str, ppid))
        process_info.SetParentProcessID(ppid);
    } else if (line.starts_with("Uid:")) {
      llvm::SmallVector<llvm::StringRef, 4> uid_parts;
      line.substr(4).trim().split(uid_parts, '\t', -1, false);
      if (uid_parts.size() >= 2) {
        uint32_t uid, euid;
        if (llvm::to_integer(uid_parts[0].trim(), uid))
          process_info.SetUserID(uid);
        if (llvm::to_integer(uid_parts[1].trim(), euid))
          process_info.SetEffectiveUserID(euid);
      }
    } else if (line.starts_with("Gid:")) {
      llvm::SmallVector<llvm::StringRef, 4> gid_parts;
      line.substr(4).trim().split(gid_parts, '\t', -1, false);
      if (gid_parts.size() >= 2) {
        uint32_t gid, egid;
        if (llvm::to_integer(gid_parts[0].trim(), gid))
          process_info.SetGroupID(gid);
        if (llvm::to_integer(gid_parts[1].trim(), egid))
          process_info.SetEffectiveGroupID(egid);
      }
    }
  }
}

// Helper function to populate command line arguments from /proc/[pid]/cmdline
void PlatformAndroid::PopulateProcessCommandLine(
    lldb::pid_t pid, ProcessInstanceInfo &process_info) {
  // Read /proc/[pid]/cmdline to get command line arguments
  Status error;
  AdbClientUP cmdline_adb = GetAdbClient(error);
  if (error.Fail())
    return;

  std::string cmdline_output;
  StreamString cmdline_cmd;
  cmdline_cmd.Printf("cat /proc/%llu/cmdline 2>/dev/null | tr '\\000' ' '",
                     static_cast<unsigned long long>(pid));
  Status cmdline_error =
      cmdline_adb->Shell(cmdline_cmd.GetData(), seconds(5), &cmdline_output);

  if (cmdline_error.Fail() || cmdline_output.empty())
    return;

  cmdline_output = llvm::StringRef(cmdline_output).trim().str();
  if (cmdline_output.empty())
    return;

  llvm::SmallVector<llvm::StringRef, 16> args;
  llvm::StringRef(cmdline_output).split(args, ' ', -1, false);
  if (args.empty())
    return;

  process_info.SetArg0(args[0]);
  Args process_args;
  for (size_t i = 1; i < args.size(); i++) {
    if (!args[i].empty())
      process_args.AppendArgument(args[i]);
  }
  process_info.SetArguments(process_args, false);
}

// Helper function to populate architecture from /proc/[pid]/exe
void PlatformAndroid::PopulateProcessArchitecture(
    lldb::pid_t pid, ProcessInstanceInfo &process_info) {
  // Read /proc/[pid]/exe to get executable path for architecture detection
  Status error;
  AdbClientUP exe_adb = GetAdbClient(error);
  if (error.Fail())
    return;

  std::string exe_output;
  StreamString exe_cmd;
  exe_cmd.Printf("readlink /proc/%llu/exe 2>/dev/null",
                 static_cast<unsigned long long>(pid));
  Status exe_error = exe_adb->Shell(exe_cmd.GetData(), seconds(5), &exe_output);

  if (exe_error.Fail() || exe_output.empty())
    return;

  exe_output = llvm::StringRef(exe_output).trim().str();

  // Determine architecture from exe path
  ArchSpec arch;
  if (exe_output.find("64") != std::string::npos ||
      exe_output.find("arm64") != std::string::npos ||
      exe_output.find("aarch64") != std::string::npos) {
    arch.SetTriple("aarch64-unknown-linux-android");
  } else if (exe_output.find("x86_64") != std::string::npos) {
    arch.SetTriple("x86_64-unknown-linux-android");
  } else if (exe_output.find("x86") != std::string::npos ||
             exe_output.find("i686") != std::string::npos) {
    arch.SetTriple("i686-unknown-linux-android");
  } else {
    // Default to armv7 for 32-bit ARM (most common on Android)
    arch.SetTriple("armv7-unknown-linux-android");
  }

  if (arch.IsValid())
    process_info.SetArchitecture(arch);
}

uint32_t
PlatformAndroid::FindProcesses(const ProcessInstanceInfoMatch &match_info,
                               ProcessInstanceInfoList &proc_infos) {
  proc_infos.clear();

  // When LLDB is running natively on an Android device (IsHost() == true),
  // use the parent class's standard Linux /proc enumeration. IsHost() is only
  // true when compiled for Android (#if defined(__ANDROID__)), so calling
  // PlatformLinux methods is safe (Android is Linux-based).
  if (IsHost())
    return PlatformLinux::FindProcesses(match_info, proc_infos);

  // Remote Android platform: implement process name lookup using 'pidof' over
  // adb.

  // LLDB stores the search name in GetExecutableFile() (even though it's
  // actually a process name like "com.android.chrome" rather than an
  // executable path). If no search name is provided, we can't use
  // 'pidof', so return early with no results.
  const ProcessInstanceInfo &match_process_info = match_info.GetProcessInfo();
  if (!match_process_info.GetExecutableFile() ||
      match_info.GetNameMatchType() == NameMatch::Ignore) {
    return 0;
  }

  // Extract the process name to search for (typically an Android package name
  // like "com.example.app" or binary name like "app_process64")
  std::string process_name = match_process_info.GetExecutableFile().GetPath();
  if (process_name.empty())
    return 0;

  // Use adb to find the process by name
  Status error;
  AdbClientUP adb(GetAdbClient(error));
  if (error.Fail()) {
    Log *log = GetLog(LLDBLog::Platform);
    LLDB_LOGF(log, "PlatformAndroid::%s failed to get ADB client: %s",
              __FUNCTION__, error.AsCString());
    return 0;
  }

  // Use 'pidof' command to get PIDs for the process name.
  // Quote the process name to handle special characters (spaces, etc.)
  std::string pidof_output;
  StreamString command;
  command.Printf("pidof '%s'", process_name.c_str());
  error = adb->Shell(command.GetData(), seconds(5), &pidof_output);

  if (error.Fail()) {
    Log *log = GetLog(LLDBLog::Platform);
    LLDB_LOG(log, "PlatformAndroid::{} 'pidof {}' failed: {}", __FUNCTION__,
             process_name.c_str(), error.AsCString());
    return 0;
  }

  // Parse PIDs from pidof output.
  // Note: pidof can return multiple PIDs (space-separated) if multiple
  // instances of the same executable are running.
  pidof_output = llvm::StringRef(pidof_output).trim().str();
  if (pidof_output.empty()) {
    Log *log = GetLog(LLDBLog::Platform);
    LLDB_LOGF(log, "PlatformAndroid::%s no process found with name '%s'",
              __FUNCTION__, process_name.c_str());
    return 0;
  }

  // Split the output by whitespace to handle multiple PIDs
  llvm::SmallVector<llvm::StringRef, 8> pid_strings;
  llvm::StringRef(pidof_output).split(pid_strings, ' ', -1, false);

  Log *log = GetLog(LLDBLog::Platform);

  // Process each PID and gather information
  uint32_t num_matches = 0;
  for (llvm::StringRef pid_str : pid_strings) {
    pid_str = pid_str.trim();
    if (pid_str.empty())
      continue;

    lldb::pid_t pid;
    if (!llvm::to_integer(pid_str, pid)) {
      LLDB_LOGF(log, "PlatformAndroid::%s failed to parse PID from: '%s'",
                __FUNCTION__, pid_str.str().c_str());
      continue;
    }

    ProcessInstanceInfo process_info;
    process_info.SetProcessID(pid);
    process_info.GetExecutableFile().SetFile(process_name,
                                             FileSpec::Style::posix);

    // Populate additional process information
    PopulateProcessStatusInfo(pid, process_info);
    PopulateProcessCommandLine(pid, process_info);
    PopulateProcessArchitecture(pid, process_info);

    // Check if this process matches the criteria
    if (match_info.Matches(process_info)) {
      proc_infos.push_back(process_info);
      num_matches++;

      LLDB_LOGF(log, "PlatformAndroid::%s found process '%s' with PID %llu",
                __FUNCTION__, process_name.c_str(),
                static_cast<unsigned long long>(pid));
    }
  }

  return num_matches;
}

std::unique_ptr<AdbSyncService> PlatformAndroid::GetSyncService(Status &error) {
  auto sync_service = std::make_unique<AdbSyncService>(m_device_id);
  error = sync_service->SetupSyncConnection();
  if (error.Fail())
    return nullptr;
  return sync_service;
}
