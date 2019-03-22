//===-- Host.mm -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Host.h"

#include <AvailabilityMacros.h>

#if !defined(MAC_OS_X_VERSION_10_7) ||                                         \
    MAC_OS_X_VERSION_MAX_ALLOWED < MAC_OS_X_VERSION_10_7
#define NO_XPC_SERVICES 1
#endif

#if !defined(NO_XPC_SERVICES)
#define __XPC_PRIVATE_H__
#include <xpc/xpc.h>

#define LaunchUsingXPCRightName "com.apple.lldb.RootDebuggingXPCService"

// These XPC messaging keys are used for communication between Host.mm and the
// XPC service.
#define LauncherXPCServiceAuthKey "auth-key"
#define LauncherXPCServiceArgPrefxKey "arg"
#define LauncherXPCServiceEnvPrefxKey "env"
#define LauncherXPCServiceCPUTypeKey "cpuType"
#define LauncherXPCServicePosixspawnFlagsKey "posixspawnFlags"
#define LauncherXPCServiceStdInPathKeyKey "stdInPath"
#define LauncherXPCServiceStdOutPathKeyKey "stdOutPath"
#define LauncherXPCServiceStdErrPathKeyKey "stdErrPath"
#define LauncherXPCServiceChildPIDKey "childPID"
#define LauncherXPCServiceErrorTypeKey "errorType"
#define LauncherXPCServiceCodeTypeKey "errorCode"

#endif

#include "llvm/Support/Host.h"

#include <asl.h>
#include <crt_externs.h>
#include <grp.h>
#include <libproc.h>
#include <pwd.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/proc.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <unistd.h>

#include "lldb/Host/ConnectionFileDescriptor.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Host/ThreadLauncher.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/CleanUp.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/NameMatches.h"
#include "lldb/Utility/ProcessInfo.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/StructuredData.h"
#include "lldb/lldb-defines.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Errno.h"

#include "../cfcpp/CFCBundle.h"
#include "../cfcpp/CFCMutableArray.h"
#include "../cfcpp/CFCMutableDictionary.h"
#include "../cfcpp/CFCReleaser.h"
#include "../cfcpp/CFCString.h"

#include <objc/objc-auto.h>

#include <CoreFoundation/CoreFoundation.h>
#include <Foundation/Foundation.h>

#ifndef _POSIX_SPAWN_DISABLE_ASLR
#define _POSIX_SPAWN_DISABLE_ASLR 0x0100
#endif

extern "C" {
int __pthread_chdir(const char *path);
int __pthread_fchdir(int fildes);
}

using namespace lldb;
using namespace lldb_private;

bool Host::GetBundleDirectory(const FileSpec &file,
                              FileSpec &bundle_directory) {
#if defined(__APPLE__)
  if (FileSystem::Instance().IsDirectory(file)) {
    char path[PATH_MAX];
    if (file.GetPath(path, sizeof(path))) {
      CFCBundle bundle(path);
      if (bundle.GetPath(path, sizeof(path))) {
        bundle_directory.SetFile(path, FileSpec::Style::native);
        return true;
      }
    }
  }
#endif
  bundle_directory.Clear();
  return false;
}

bool Host::ResolveExecutableInBundle(FileSpec &file) {
#if defined(__APPLE__)
  if (FileSystem::Instance().IsDirectory(file)) {
    char path[PATH_MAX];
    if (file.GetPath(path, sizeof(path))) {
      CFCBundle bundle(path);
      CFCReleaser<CFURLRef> url(bundle.CopyExecutableURL());
      if (url.get()) {
        if (::CFURLGetFileSystemRepresentation(url.get(), YES, (UInt8 *)path,
                                               sizeof(path))) {
          file.SetFile(path, FileSpec::Style::native);
          return true;
        }
      }
    }
  }
#endif
  return false;
}

static void *AcceptPIDFromInferior(void *arg) {
  const char *connect_url = (const char *)arg;
  ConnectionFileDescriptor file_conn;
  Status error;
  if (file_conn.Connect(connect_url, &error) == eConnectionStatusSuccess) {
    char pid_str[256];
    ::memset(pid_str, 0, sizeof(pid_str));
    ConnectionStatus status;
    const size_t pid_str_len = file_conn.Read(
        pid_str, sizeof(pid_str), std::chrono::seconds(0), status, NULL);
    if (pid_str_len > 0) {
      int pid = atoi(pid_str);
      return (void *)(intptr_t)pid;
    }
  }
  return NULL;
}

static bool WaitForProcessToSIGSTOP(const lldb::pid_t pid,
                                    const int timeout_in_seconds) {
  const int time_delta_usecs = 100000;
  const int num_retries = timeout_in_seconds / time_delta_usecs;
  for (int i = 0; i < num_retries; i++) {
    struct proc_bsdinfo bsd_info;
    int error = ::proc_pidinfo(pid, PROC_PIDTBSDINFO, (uint64_t)0, &bsd_info,
                               PROC_PIDTBSDINFO_SIZE);

    switch (error) {
    case EINVAL:
    case ENOTSUP:
    case ESRCH:
    case EPERM:
      return false;

    default:
      break;

    case 0:
      if (bsd_info.pbi_status == SSTOP)
        return true;
    }
    ::usleep(time_delta_usecs);
  }
  return false;
}
#if !defined(__arm__) && !defined(__arm64__) && !defined(__aarch64__)

const char *applscript_in_new_tty = "tell application \"Terminal\"\n"
                                    "   activate\n"
                                    "	do script \"/bin/bash -c '%s';exit\"\n"
                                    "end tell\n";

const char *applscript_in_existing_tty = "\
set the_shell_script to \"/bin/bash -c '%s';exit\"\n\
tell application \"Terminal\"\n\
	repeat with the_window in (get windows)\n\
		repeat with the_tab in tabs of the_window\n\
			set the_tty to tty in the_tab\n\
			if the_tty contains \"%s\" then\n\
				if the_tab is not busy then\n\
					set selected of the_tab to true\n\
					set frontmost of the_window to true\n\
					do script the_shell_script in the_tab\n\
					return\n\
				end if\n\
			end if\n\
		end repeat\n\
	end repeat\n\
	do script the_shell_script\n\
end tell\n";

static Status
LaunchInNewTerminalWithAppleScript(const char *exe_path,
                                   ProcessLaunchInfo &launch_info) {
  Status error;
  char unix_socket_name[PATH_MAX] = "/tmp/XXXXXX";
  if (::mktemp(unix_socket_name) == NULL) {
    error.SetErrorString("failed to make temporary path for a unix socket");
    return error;
  }

  StreamString command;
  FileSpec darwin_debug_file_spec = HostInfo::GetSupportExeDir();
  if (!darwin_debug_file_spec) {
    error.SetErrorString("can't locate the 'darwin-debug' executable");
    return error;
  }

  darwin_debug_file_spec.GetFilename().SetCString("darwin-debug");

  if (!FileSystem::Instance().Exists(darwin_debug_file_spec)) {
    error.SetErrorStringWithFormat(
        "the 'darwin-debug' executable doesn't exists at '%s'",
        darwin_debug_file_spec.GetPath().c_str());
    return error;
  }

  char launcher_path[PATH_MAX];
  darwin_debug_file_spec.GetPath(launcher_path, sizeof(launcher_path));

  const ArchSpec &arch_spec = launch_info.GetArchitecture();
  // Only set the architecture if it is valid and if it isn't Haswell (x86_64h).
  if (arch_spec.IsValid() &&
      arch_spec.GetCore() != ArchSpec::eCore_x86_64_x86_64h)
    command.Printf("arch -arch %s ", arch_spec.GetArchitectureName());

  command.Printf("'%s' --unix-socket=%s", launcher_path, unix_socket_name);

  if (arch_spec.IsValid())
    command.Printf(" --arch=%s", arch_spec.GetArchitectureName());

  FileSpec working_dir{launch_info.GetWorkingDirectory()};
  if (working_dir)
    command.Printf(" --working-dir '%s'", working_dir.GetCString());
  else {
    char cwd[PATH_MAX];
    if (getcwd(cwd, PATH_MAX))
      command.Printf(" --working-dir '%s'", cwd);
  }

  if (launch_info.GetFlags().Test(eLaunchFlagDisableASLR))
    command.PutCString(" --disable-aslr");

  // We are launching on this host in a terminal. So compare the environment on
  // the host to what is supplied in the launch_info. Any items that aren't in
  // the host environment need to be sent to darwin-debug. If we send all
  // environment entries, we might blow the max command line length, so we only
  // send user modified entries.
  Environment host_env = Host::GetEnvironment();

  for (const auto &KV : launch_info.GetEnvironment()) {
    auto host_entry = host_env.find(KV.first());
    if (host_entry == host_env.end() || host_entry->second != KV.second)
      command.Format(" --env='{0}'", Environment::compose(KV));
  }

  command.PutCString(" -- ");

  const char **argv = launch_info.GetArguments().GetConstArgumentVector();
  if (argv) {
    for (size_t i = 0; argv[i] != NULL; ++i) {
      if (i == 0)
        command.Printf(" '%s'", exe_path);
      else
        command.Printf(" '%s'", argv[i]);
    }
  } else {
    command.Printf(" '%s'", exe_path);
  }
  command.PutCString(" ; echo Process exited with status $?");
  if (launch_info.GetFlags().Test(lldb::eLaunchFlagCloseTTYOnExit))
    command.PutCString(" ; exit");

  StreamString applescript_source;

  applescript_source.Printf(applscript_in_new_tty,
                            command.GetString().str().c_str());
  NSAppleScript *applescript = [[NSAppleScript alloc]
      initWithSource:[NSString stringWithCString:applescript_source.GetString()
                                                     .str()
                                                     .c_str()
                                        encoding:NSUTF8StringEncoding]];

  lldb::pid_t pid = LLDB_INVALID_PROCESS_ID;

  Status lldb_error;
  // Sleep and wait a bit for debugserver to start to listen...
  ConnectionFileDescriptor file_conn;
  char connect_url[128];
  ::snprintf(connect_url, sizeof(connect_url), "unix-accept://%s",
             unix_socket_name);

  // Spawn a new thread to accept incoming connection on the connect_url
  // so we can grab the pid from the inferior. We have to do this because we
  // are sending an AppleScript that will launch a process in Terminal.app,
  // in a shell and the shell will fork/exec a couple of times before we get
  // to the process that we wanted to launch. So when our process actually
  // gets launched, we will handshake with it and get the process ID for it.
  HostThread accept_thread = ThreadLauncher::LaunchThread(
      unix_socket_name, AcceptPIDFromInferior, connect_url, &lldb_error);

  [applescript executeAndReturnError:nil];

  thread_result_t accept_thread_result = NULL;
  lldb_error = accept_thread.Join(&accept_thread_result);
  if (lldb_error.Success() && accept_thread_result) {
    pid = (intptr_t)accept_thread_result;

    // Wait for process to be stopped at the entry point by watching
    // for the process status to be set to SSTOP which indicates it it
    // SIGSTOP'ed at the entry point
    WaitForProcessToSIGSTOP(pid, 5);
  }

  llvm::sys::fs::remove(unix_socket_name);
  [applescript release];
  if (pid != LLDB_INVALID_PROCESS_ID)
    launch_info.SetProcessID(pid);
  return error;
}

#endif // #if !defined(__arm__) && !defined(__arm64__) && !defined(__aarch64__)

bool Host::OpenFileInExternalEditor(const FileSpec &file_spec,
                                    uint32_t line_no) {
#if defined(__arm__) || defined(__arm64__) || defined(__aarch64__)
  return false;
#else
  // We attach this to an 'odoc' event to specify a particular selection
  typedef struct {
    int16_t reserved0; // must be zero
    int16_t fLineNumber;
    int32_t fSelStart;
    int32_t fSelEnd;
    uint32_t reserved1; // must be zero
    uint32_t reserved2; // must be zero
  } BabelAESelInfo;

  Log *log(lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_HOST));
  char file_path[PATH_MAX];
  file_spec.GetPath(file_path, PATH_MAX);
  CFCString file_cfstr(file_path, kCFStringEncodingUTF8);
  CFCReleaser<CFURLRef> file_URL(::CFURLCreateWithFileSystemPath(
      NULL, file_cfstr.get(), kCFURLPOSIXPathStyle, false));

  if (log)
    log->Printf(
        "Sending source file: \"%s\" and line: %d to external editor.\n",
        file_path, line_no);

  long error;
  BabelAESelInfo file_and_line_info = {
      0,                      // reserved0
      (int16_t)(line_no - 1), // fLineNumber (zero based line number)
      1,                      // fSelStart
      1024,                   // fSelEnd
      0,                      // reserved1
      0                       // reserved2
  };

  AEKeyDesc file_and_line_desc;

  error = ::AECreateDesc(typeUTF8Text, &file_and_line_info,
                         sizeof(file_and_line_info),
                         &(file_and_line_desc.descContent));

  if (error != noErr) {
    if (log)
      log->Printf("Error creating AEDesc: %ld.\n", error);
    return false;
  }

  file_and_line_desc.descKey = keyAEPosition;

  static std::string g_app_name;
  static FSRef g_app_fsref;

  LSApplicationParameters app_params;
  ::memset(&app_params, 0, sizeof(app_params));
  app_params.flags =
      kLSLaunchDefaults | kLSLaunchDontAddToRecents | kLSLaunchDontSwitch;

  char *external_editor = ::getenv("LLDB_EXTERNAL_EDITOR");

  if (external_editor) {
    if (log)
      log->Printf("Looking for external editor \"%s\".\n", external_editor);

    if (g_app_name.empty() ||
        strcmp(g_app_name.c_str(), external_editor) != 0) {
      CFCString editor_name(external_editor, kCFStringEncodingUTF8);
      error = ::LSFindApplicationForInfo(kLSUnknownCreator, NULL,
                                         editor_name.get(), &g_app_fsref, NULL);

      // If we found the app, then store away the name so we don't have to
      // re-look it up.
      if (error != noErr) {
        if (log)
          log->Printf(
              "Could not find External Editor application, error: %ld.\n",
              error);
        return false;
      }
    }
    app_params.application = &g_app_fsref;
  }

  ProcessSerialNumber psn;
  CFCReleaser<CFArrayRef> file_array(
      CFArrayCreate(NULL, (const void **)file_URL.ptr_address(false), 1, NULL));
  error = ::LSOpenURLsWithRole(file_array.get(), kLSRolesAll,
                               &file_and_line_desc, &app_params, &psn, 1);

  AEDisposeDesc(&(file_and_line_desc.descContent));

  if (error != noErr) {
    if (log)
      log->Printf("LSOpenURLsWithRole failed, error: %ld.\n", error);

    return false;
  }

  return true;
#endif // #if !defined(__arm__) && !defined(__arm64__) && !defined(__aarch64__)
}

Environment Host::GetEnvironment() { return Environment(*_NSGetEnviron()); }

static bool GetMacOSXProcessCPUType(ProcessInstanceInfo &process_info) {
  if (process_info.ProcessIDIsValid()) {
    // Make a new mib to stay thread safe
    int mib[CTL_MAXNAME] = {
        0,
    };
    size_t mib_len = CTL_MAXNAME;
    if (::sysctlnametomib("sysctl.proc_cputype", mib, &mib_len))
      return false;

    mib[mib_len] = process_info.GetProcessID();
    mib_len++;

    cpu_type_t cpu, sub = 0;
    size_t len = sizeof(cpu);
    if (::sysctl(mib, mib_len, &cpu, &len, 0, 0) == 0) {
      switch (cpu) {
      case CPU_TYPE_I386:
        sub = CPU_SUBTYPE_I386_ALL;
        break;
      case CPU_TYPE_X86_64:
        sub = CPU_SUBTYPE_X86_64_ALL;
        break;

#if defined(CPU_TYPE_ARM64) && defined(CPU_SUBTYPE_ARM64_ALL)
      case CPU_TYPE_ARM64:
        sub = CPU_SUBTYPE_ARM64_ALL;
        break;
#endif

      case CPU_TYPE_ARM: {
        // Note that we fetched the cpu type from the PROCESS but we can't get a
        // cpusubtype of the
        // process -- we can only get the host's cpu subtype.
        uint32_t cpusubtype = 0;
        len = sizeof(cpusubtype);
        if (::sysctlbyname("hw.cpusubtype", &cpusubtype, &len, NULL, 0) == 0)
          sub = cpusubtype;

        bool host_cpu_is_64bit;
        uint32_t is64bit_capable;
        size_t is64bit_capable_len = sizeof(is64bit_capable);
        host_cpu_is_64bit =
            sysctlbyname("hw.cpu64bit_capable", &is64bit_capable,
                         &is64bit_capable_len, NULL, 0) == 0;

        // if the host is an armv8 device, its cpusubtype will be in
        // CPU_SUBTYPE_ARM64 numbering
        // and we need to rewrite it to a reasonable CPU_SUBTYPE_ARM value
        // instead.

        if (host_cpu_is_64bit) {
          sub = CPU_SUBTYPE_ARM_V7;
        }
      } break;

      default:
        break;
      }
      process_info.GetArchitecture().SetArchitecture(eArchTypeMachO, cpu, sub);
      return true;
    }
  }
  process_info.GetArchitecture().Clear();
  return false;
}

static bool GetMacOSXProcessArgs(const ProcessInstanceInfoMatch *match_info_ptr,
                                 ProcessInstanceInfo &process_info) {
  if (process_info.ProcessIDIsValid()) {
    int proc_args_mib[3] = {CTL_KERN, KERN_PROCARGS2,
                            (int)process_info.GetProcessID()};

    size_t arg_data_size = 0;
    if (::sysctl(proc_args_mib, 3, nullptr, &arg_data_size, NULL, 0) ||
        arg_data_size == 0)
      arg_data_size = 8192;

    // Add a few bytes to the calculated length, I know we need to add at least
    // one byte
    // to this number otherwise we get junk back, so add 128 just in case...
    DataBufferHeap arg_data(arg_data_size + 128, 0);
    arg_data_size = arg_data.GetByteSize();
    if (::sysctl(proc_args_mib, 3, arg_data.GetBytes(), &arg_data_size, NULL,
                 0) == 0) {
      DataExtractor data(arg_data.GetBytes(), arg_data_size,
                         endian::InlHostByteOrder(), sizeof(void *));
      lldb::offset_t offset = 0;
      uint32_t argc = data.GetU32(&offset);
      llvm::Triple &triple = process_info.GetArchitecture().GetTriple();
      const llvm::Triple::ArchType triple_arch = triple.getArch();
      const bool check_for_ios_simulator =
          (triple_arch == llvm::Triple::x86 ||
           triple_arch == llvm::Triple::x86_64);
      const char *cstr = data.GetCStr(&offset);
      if (cstr) {
        process_info.GetExecutableFile().SetFile(cstr, FileSpec::Style::native);

        if (match_info_ptr == NULL ||
            NameMatches(
                process_info.GetExecutableFile().GetFilename().GetCString(),
                match_info_ptr->GetNameMatchType(),
                match_info_ptr->GetProcessInfo().GetName())) {
          // Skip NULLs
          while (1) {
            const uint8_t *p = data.PeekData(offset, 1);
            if ((p == NULL) || (*p != '\0'))
              break;
            ++offset;
          }
          // Now extract all arguments
          Args &proc_args = process_info.GetArguments();
          for (int i = 0; i < static_cast<int>(argc); ++i) {
            cstr = data.GetCStr(&offset);
            if (cstr)
              proc_args.AppendArgument(llvm::StringRef(cstr));
          }

          Environment &proc_env = process_info.GetEnvironment();
          while ((cstr = data.GetCStr(&offset))) {
            if (cstr[0] == '\0')
              break;

            if (check_for_ios_simulator) {
              if (strncmp(cstr, "SIMULATOR_UDID=", strlen("SIMULATOR_UDID=")) ==
                  0)
                process_info.GetArchitecture().GetTriple().setOS(
                    llvm::Triple::IOS);
              else
                process_info.GetArchitecture().GetTriple().setOS(
                    llvm::Triple::MacOSX);
            }

            proc_env.insert(cstr);
          }
          return true;
        }
      }
    }
  }
  return false;
}

static bool GetMacOSXProcessUserAndGroup(ProcessInstanceInfo &process_info) {
  if (process_info.ProcessIDIsValid()) {
    int mib[4];
    mib[0] = CTL_KERN;
    mib[1] = KERN_PROC;
    mib[2] = KERN_PROC_PID;
    mib[3] = process_info.GetProcessID();
    struct kinfo_proc proc_kinfo;
    size_t proc_kinfo_size = sizeof(struct kinfo_proc);

    if (::sysctl(mib, 4, &proc_kinfo, &proc_kinfo_size, NULL, 0) == 0) {
      if (proc_kinfo_size > 0) {
        process_info.SetParentProcessID(proc_kinfo.kp_eproc.e_ppid);
        process_info.SetUserID(proc_kinfo.kp_eproc.e_pcred.p_ruid);
        process_info.SetGroupID(proc_kinfo.kp_eproc.e_pcred.p_rgid);
        process_info.SetEffectiveUserID(proc_kinfo.kp_eproc.e_ucred.cr_uid);
        if (proc_kinfo.kp_eproc.e_ucred.cr_ngroups > 0)
          process_info.SetEffectiveGroupID(
              proc_kinfo.kp_eproc.e_ucred.cr_groups[0]);
        else
          process_info.SetEffectiveGroupID(UINT32_MAX);
        return true;
      }
    }
  }
  process_info.SetParentProcessID(LLDB_INVALID_PROCESS_ID);
  process_info.SetUserID(UINT32_MAX);
  process_info.SetGroupID(UINT32_MAX);
  process_info.SetEffectiveUserID(UINT32_MAX);
  process_info.SetEffectiveGroupID(UINT32_MAX);
  return false;
}

uint32_t Host::FindProcesses(const ProcessInstanceInfoMatch &match_info,
                             ProcessInstanceInfoList &process_infos) {
  std::vector<struct kinfo_proc> kinfos;

  int mib[3] = {CTL_KERN, KERN_PROC, KERN_PROC_ALL};

  size_t pid_data_size = 0;
  if (::sysctl(mib, 3, nullptr, &pid_data_size, nullptr, 0) != 0)
    return 0;

  // Add a few extra in case a few more show up
  const size_t estimated_pid_count =
      (pid_data_size / sizeof(struct kinfo_proc)) + 10;

  kinfos.resize(estimated_pid_count);
  pid_data_size = kinfos.size() * sizeof(struct kinfo_proc);

  if (::sysctl(mib, 3, &kinfos[0], &pid_data_size, nullptr, 0) != 0)
    return 0;

  const size_t actual_pid_count = (pid_data_size / sizeof(struct kinfo_proc));

  bool all_users = match_info.GetMatchAllUsers();
  const lldb::pid_t our_pid = getpid();
  const uid_t our_uid = getuid();
  for (size_t i = 0; i < actual_pid_count; i++) {
    const struct kinfo_proc &kinfo = kinfos[i];

    bool kinfo_user_matches = false;
    if (all_users)
      kinfo_user_matches = true;
    else
      kinfo_user_matches = kinfo.kp_eproc.e_pcred.p_ruid == our_uid;

    // Special case, if lldb is being run as root we can attach to anything.
    if (our_uid == 0)
      kinfo_user_matches = true;

    if (!kinfo_user_matches || // Make sure the user is acceptable
        static_cast<lldb::pid_t>(kinfo.kp_proc.p_pid) ==
            our_pid ||                   // Skip this process
        kinfo.kp_proc.p_pid == 0 ||      // Skip kernel (kernel pid is zero)
        kinfo.kp_proc.p_stat == SZOMB || // Zombies are bad, they like brains...
        kinfo.kp_proc.p_flag & P_TRACED ||   // Being debugged?
        kinfo.kp_proc.p_flag & P_WEXIT ||    // Working on exiting?
        kinfo.kp_proc.p_flag & P_TRANSLATED) // Skip translated ppc (Rosetta)
      continue;

    ProcessInstanceInfo process_info;
    process_info.SetProcessID(kinfo.kp_proc.p_pid);
    process_info.SetParentProcessID(kinfo.kp_eproc.e_ppid);
    process_info.SetUserID(kinfo.kp_eproc.e_pcred.p_ruid);
    process_info.SetGroupID(kinfo.kp_eproc.e_pcred.p_rgid);
    process_info.SetEffectiveUserID(kinfo.kp_eproc.e_ucred.cr_uid);
    if (kinfo.kp_eproc.e_ucred.cr_ngroups > 0)
      process_info.SetEffectiveGroupID(kinfo.kp_eproc.e_ucred.cr_groups[0]);
    else
      process_info.SetEffectiveGroupID(UINT32_MAX);

    // Make sure our info matches before we go fetch the name and cpu type
    if (match_info.Matches(process_info)) {
      // Get CPU type first so we can know to look for iOS simulator is we have
      // x86 or x86_64
      if (GetMacOSXProcessCPUType(process_info)) {
        if (GetMacOSXProcessArgs(&match_info, process_info)) {
          if (match_info.Matches(process_info))
            process_infos.Append(process_info);
        }
      }
    }
  }
  return process_infos.GetSize();
}

bool Host::GetProcessInfo(lldb::pid_t pid, ProcessInstanceInfo &process_info) {
  process_info.SetProcessID(pid);
  bool success = false;

  // Get CPU type first so we can know to look for iOS simulator is we have x86
  // or x86_64
  if (GetMacOSXProcessCPUType(process_info))
    success = true;

  if (GetMacOSXProcessArgs(NULL, process_info))
    success = true;

  if (GetMacOSXProcessUserAndGroup(process_info))
    success = true;

  if (success)
    return true;

  process_info.Clear();
  return false;
}

#if !NO_XPC_SERVICES
static void PackageXPCArguments(xpc_object_t message, const char *prefix,
                                const Args &args) {
  size_t count = args.GetArgumentCount();
  char buf[50]; // long enough for 'argXXX'
  memset(buf, 0, 50);
  sprintf(buf, "%sCount", prefix);
  xpc_dictionary_set_int64(message, buf, count);
  for (size_t i = 0; i < count; i++) {
    memset(buf, 0, 50);
    sprintf(buf, "%s%zi", prefix, i);
    xpc_dictionary_set_string(message, buf, args.GetArgumentAtIndex(i));
  }
}

static void PackageXPCEnvironment(xpc_object_t message, llvm::StringRef prefix,
                                  const Environment &env) {
  xpc_dictionary_set_int64(message, (prefix + "Count").str().c_str(),
                           env.size());
  size_t i = 0;
  for (const auto &KV : env) {
    xpc_dictionary_set_string(message, (prefix + llvm::Twine(i)).str().c_str(),
                              Environment::compose(KV).c_str());
  }
}

/*
 A valid authorizationRef means that
    - there is the LaunchUsingXPCRightName rights in the /etc/authorization
    - we have successfully copied the rights to be send over the XPC wire
 Once obtained, it will be valid for as long as the process lives.
 */
static AuthorizationRef authorizationRef = NULL;
static Status getXPCAuthorization(ProcessLaunchInfo &launch_info) {
  Status error;
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST |
                                                  LIBLLDB_LOG_PROCESS));

  if ((launch_info.GetUserID() == 0) && !authorizationRef) {
    OSStatus createStatus =
        AuthorizationCreate(NULL, kAuthorizationEmptyEnvironment,
                            kAuthorizationFlagDefaults, &authorizationRef);
    if (createStatus != errAuthorizationSuccess) {
      error.SetError(1, eErrorTypeGeneric);
      error.SetErrorString("Can't create authorizationRef.");
      LLDB_LOG(log, "error: {0}", error);
      return error;
    }

    OSStatus rightsStatus =
        AuthorizationRightGet(LaunchUsingXPCRightName, NULL);
    if (rightsStatus != errAuthorizationSuccess) {
      // No rights in the security database, Create it with the right prompt.
      CFStringRef prompt =
          CFSTR("Xcode is trying to take control of a root process.");
      CFStringRef keys[] = {CFSTR("en")};
      CFTypeRef values[] = {prompt};
      CFDictionaryRef promptDict = CFDictionaryCreate(
          kCFAllocatorDefault, (const void **)keys, (const void **)values, 1,
          &kCFCopyStringDictionaryKeyCallBacks,
          &kCFTypeDictionaryValueCallBacks);

      CFStringRef keys1[] = {CFSTR("class"), CFSTR("group"), CFSTR("comment"),
                             CFSTR("default-prompt"), CFSTR("shared")};
      CFTypeRef values1[] = {CFSTR("user"), CFSTR("admin"),
                             CFSTR(LaunchUsingXPCRightName), promptDict,
                             kCFBooleanFalse};
      CFDictionaryRef dict = CFDictionaryCreate(
          kCFAllocatorDefault, (const void **)keys1, (const void **)values1, 5,
          &kCFCopyStringDictionaryKeyCallBacks,
          &kCFTypeDictionaryValueCallBacks);
      rightsStatus = AuthorizationRightSet(
          authorizationRef, LaunchUsingXPCRightName, dict, NULL, NULL, NULL);
      CFRelease(promptDict);
      CFRelease(dict);
    }

    OSStatus copyRightStatus = errAuthorizationDenied;
    if (rightsStatus == errAuthorizationSuccess) {
      AuthorizationItem item1 = {LaunchUsingXPCRightName, 0, NULL, 0};
      AuthorizationItem items[] = {item1};
      AuthorizationRights requestedRights = {1, items};
      AuthorizationFlags authorizationFlags =
          kAuthorizationFlagInteractionAllowed | kAuthorizationFlagExtendRights;
      copyRightStatus = AuthorizationCopyRights(
          authorizationRef, &requestedRights, kAuthorizationEmptyEnvironment,
          authorizationFlags, NULL);
    }

    if (copyRightStatus != errAuthorizationSuccess) {
      // Eventually when the commandline supports running as root and the user
      // is not
      // logged in in the current audit session, we will need the trick in gdb
      // where
      // we ask the user to type in the root passwd in the terminal.
      error.SetError(2, eErrorTypeGeneric);
      error.SetErrorStringWithFormat(
          "Launching as root needs root authorization.");
      LLDB_LOG(log, "error: {0}", error);

      if (authorizationRef) {
        AuthorizationFree(authorizationRef, kAuthorizationFlagDefaults);
        authorizationRef = NULL;
      }
    }
  }

  return error;
}
#endif

static short GetPosixspawnFlags(const ProcessLaunchInfo &launch_info) {
  short flags = POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK;

  if (launch_info.GetFlags().Test(eLaunchFlagExec))
    flags |= POSIX_SPAWN_SETEXEC; // Darwin specific posix_spawn flag

  if (launch_info.GetFlags().Test(eLaunchFlagDebug))
    flags |= POSIX_SPAWN_START_SUSPENDED; // Darwin specific posix_spawn flag

  if (launch_info.GetFlags().Test(eLaunchFlagDisableASLR))
    flags |= _POSIX_SPAWN_DISABLE_ASLR; // Darwin specific posix_spawn flag

  if (launch_info.GetLaunchInSeparateProcessGroup())
    flags |= POSIX_SPAWN_SETPGROUP;

#ifdef POSIX_SPAWN_CLOEXEC_DEFAULT
#if defined(__x86_64__) || defined(__i386__)
  static LazyBool g_use_close_on_exec_flag = eLazyBoolCalculate;
  if (g_use_close_on_exec_flag == eLazyBoolCalculate) {
    g_use_close_on_exec_flag = eLazyBoolNo;

    llvm::VersionTuple version = HostInfo::GetOSVersion();
    if (version > llvm::VersionTuple(10, 7)) {
      // Kernel panic if we use the POSIX_SPAWN_CLOEXEC_DEFAULT on 10.7 or
      // earlier
      g_use_close_on_exec_flag = eLazyBoolYes;
    }
  }
#else
  static LazyBool g_use_close_on_exec_flag = eLazyBoolYes;
#endif // defined(__x86_64__) || defined(__i386__)
  // Close all files exception those with file actions if this is supported.
  if (g_use_close_on_exec_flag == eLazyBoolYes)
    flags |= POSIX_SPAWN_CLOEXEC_DEFAULT;
#endif // ifdef POSIX_SPAWN_CLOEXEC_DEFAULT
  return flags;
}

static Status LaunchProcessXPC(const char *exe_path,
                               ProcessLaunchInfo &launch_info,
                               lldb::pid_t &pid) {
#if !NO_XPC_SERVICES
  Status error = getXPCAuthorization(launch_info);
  if (error.Fail())
    return error;

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST |
                                                  LIBLLDB_LOG_PROCESS));

  uid_t requested_uid = launch_info.GetUserID();
  const char *xpc_service = nil;
  bool send_auth = false;
  AuthorizationExternalForm extForm;
  if (requested_uid == 0) {
    if (AuthorizationMakeExternalForm(authorizationRef, &extForm) ==
        errAuthorizationSuccess) {
      send_auth = true;
    } else {
      error.SetError(3, eErrorTypeGeneric);
      error.SetErrorStringWithFormat("Launching root via XPC needs to "
                                     "externalize authorization reference.");
      LLDB_LOG(log, "error: {0}", error);
      return error;
    }
    xpc_service = LaunchUsingXPCRightName;
  } else {
    error.SetError(4, eErrorTypeGeneric);
    error.SetErrorStringWithFormat(
        "Launching via XPC is only currently available for root.");
    LLDB_LOG(log, "error: {0}", error);
    return error;
  }

  xpc_connection_t conn = xpc_connection_create(xpc_service, NULL);

  xpc_connection_set_event_handler(conn, ^(xpc_object_t event) {
    xpc_type_t type = xpc_get_type(event);

    if (type == XPC_TYPE_ERROR) {
      if (event == XPC_ERROR_CONNECTION_INTERRUPTED) {
        // The service has either canceled itself, crashed, or been terminated.
        // The XPC connection is still valid and sending a message to it will
        // re-launch the service.
        // If the service is state-full, this is the time to initialize the new
        // service.
        return;
      } else if (event == XPC_ERROR_CONNECTION_INVALID) {
        // The service is invalid. Either the service name supplied to
        // xpc_connection_create() is incorrect
        // or we (this process) have canceled the service; we can do any cleanup
        // of application state at this point.
        // printf("Service disconnected");
        return;
      } else {
        // printf("Unexpected error from service: %s",
        // xpc_dictionary_get_string(event, XPC_ERROR_KEY_DESCRIPTION));
      }

    } else {
      // printf("Received unexpected event in handler");
    }
  });

  xpc_connection_set_finalizer_f(conn, xpc_finalizer_t(xpc_release));
  xpc_connection_resume(conn);
  xpc_object_t message = xpc_dictionary_create(nil, nil, 0);

  if (send_auth) {
    xpc_dictionary_set_data(message, LauncherXPCServiceAuthKey, extForm.bytes,
                            sizeof(AuthorizationExternalForm));
  }

  PackageXPCArguments(message, LauncherXPCServiceArgPrefxKey,
                      launch_info.GetArguments());
  PackageXPCEnvironment(message, LauncherXPCServiceEnvPrefxKey,
                        launch_info.GetEnvironment());

  // Posix spawn stuff.
  xpc_dictionary_set_int64(message, LauncherXPCServiceCPUTypeKey,
                           launch_info.GetArchitecture().GetMachOCPUType());
  xpc_dictionary_set_int64(message, LauncherXPCServicePosixspawnFlagsKey,
                           GetPosixspawnFlags(launch_info));
  const FileAction *file_action = launch_info.GetFileActionForFD(STDIN_FILENO);
  if (file_action && !file_action->GetPath().empty()) {
    xpc_dictionary_set_string(message, LauncherXPCServiceStdInPathKeyKey,
                              file_action->GetPath().str().c_str());
  }
  file_action = launch_info.GetFileActionForFD(STDOUT_FILENO);
  if (file_action && !file_action->GetPath().empty()) {
    xpc_dictionary_set_string(message, LauncherXPCServiceStdOutPathKeyKey,
                              file_action->GetPath().str().c_str());
  }
  file_action = launch_info.GetFileActionForFD(STDERR_FILENO);
  if (file_action && !file_action->GetPath().empty()) {
    xpc_dictionary_set_string(message, LauncherXPCServiceStdErrPathKeyKey,
                              file_action->GetPath().str().c_str());
  }

  xpc_object_t reply =
      xpc_connection_send_message_with_reply_sync(conn, message);
  xpc_type_t returnType = xpc_get_type(reply);
  if (returnType == XPC_TYPE_DICTIONARY) {
    pid = xpc_dictionary_get_int64(reply, LauncherXPCServiceChildPIDKey);
    if (pid == 0) {
      int errorType =
          xpc_dictionary_get_int64(reply, LauncherXPCServiceErrorTypeKey);
      int errorCode =
          xpc_dictionary_get_int64(reply, LauncherXPCServiceCodeTypeKey);

      error.SetError(errorCode, eErrorTypeGeneric);
      error.SetErrorStringWithFormat(
          "Problems with launching via XPC. Error type : %i, code : %i",
          errorType, errorCode);
      LLDB_LOG(log, "error: {0}", error);

      if (authorizationRef) {
        AuthorizationFree(authorizationRef, kAuthorizationFlagDefaults);
        authorizationRef = NULL;
      }
    }
  } else if (returnType == XPC_TYPE_ERROR) {
    error.SetError(5, eErrorTypeGeneric);
    error.SetErrorStringWithFormat(
        "Problems with launching via XPC. XPC error : %s",
        xpc_dictionary_get_string(reply, XPC_ERROR_KEY_DESCRIPTION));
    LLDB_LOG(log, "error: {0}", error);
  }

  return error;
#else
  Status error;
  return error;
#endif
}

static bool AddPosixSpawnFileAction(void *_file_actions, const FileAction *info,
                                    Log *log, Status &error) {
  if (info == NULL)
    return false;

  posix_spawn_file_actions_t *file_actions =
      reinterpret_cast<posix_spawn_file_actions_t *>(_file_actions);

  switch (info->GetAction()) {
  case FileAction::eFileActionNone:
    error.Clear();
    break;

  case FileAction::eFileActionClose:
    if (info->GetFD() == -1)
      error.SetErrorString(
          "invalid fd for posix_spawn_file_actions_addclose(...)");
    else {
      error.SetError(
          ::posix_spawn_file_actions_addclose(file_actions, info->GetFD()),
          eErrorTypePOSIX);
      if (error.Fail())
        LLDB_LOG(log,
                 "error: {0}, posix_spawn_file_actions_addclose "
                 "(action={1}, fd={2})",
                 error, file_actions, info->GetFD());
    }
    break;

  case FileAction::eFileActionDuplicate:
    if (info->GetFD() == -1)
      error.SetErrorString(
          "invalid fd for posix_spawn_file_actions_adddup2(...)");
    else if (info->GetActionArgument() == -1)
      error.SetErrorString(
          "invalid duplicate fd for posix_spawn_file_actions_adddup2(...)");
    else {
      error.SetError(
          ::posix_spawn_file_actions_adddup2(file_actions, info->GetFD(),
                                             info->GetActionArgument()),
          eErrorTypePOSIX);
      if (error.Fail())
        LLDB_LOG(log,
                 "error: {0}, posix_spawn_file_actions_adddup2 "
                 "(action={1}, fd={2}, dup_fd={3})",
                 error, file_actions, info->GetFD(), info->GetActionArgument());
    }
    break;

  case FileAction::eFileActionOpen:
    if (info->GetFD() == -1)
      error.SetErrorString(
          "invalid fd in posix_spawn_file_actions_addopen(...)");
    else {
      int oflag = info->GetActionArgument();

      mode_t mode = 0;

      if (oflag & O_CREAT)
        mode = 0640;

      error.SetError(::posix_spawn_file_actions_addopen(
                         file_actions, info->GetFD(),
                         info->GetPath().str().c_str(), oflag, mode),
                     eErrorTypePOSIX);
      if (error.Fail())
        LLDB_LOG(log,
                 "error: {0}, posix_spawn_file_actions_addopen (action={1}, "
                 "fd={2}, path='{3}', oflag={4}, mode={5})",
                 error, file_actions, info->GetFD(), info->GetPath(), oflag,
                 mode);
    }
    break;
  }
  return error.Success();
}

static Status LaunchProcessPosixSpawn(const char *exe_path,
                                      const ProcessLaunchInfo &launch_info,
                                      lldb::pid_t &pid) {
  Status error;
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST |
                                                  LIBLLDB_LOG_PROCESS));

  posix_spawnattr_t attr;
  error.SetError(::posix_spawnattr_init(&attr), eErrorTypePOSIX);

  if (error.Fail()) {
    LLDB_LOG(log, "error: {0}, ::posix_spawnattr_init ( &attr )", error);
    return error;
  }

  // Make sure we clean up the posix spawn attributes before exiting this scope.
  CleanUp cleanup_attr(posix_spawnattr_destroy, &attr);

  sigset_t no_signals;
  sigset_t all_signals;
  sigemptyset(&no_signals);
  sigfillset(&all_signals);
  ::posix_spawnattr_setsigmask(&attr, &no_signals);
  ::posix_spawnattr_setsigdefault(&attr, &all_signals);

  short flags = GetPosixspawnFlags(launch_info);

  error.SetError(::posix_spawnattr_setflags(&attr, flags), eErrorTypePOSIX);
  if (error.Fail()) {
    LLDB_LOG(log,
             "error: {0}, ::posix_spawnattr_setflags ( &attr, flags={1:x} )",
             error, flags);
    return error;
  }

// posix_spawnattr_setbinpref_np appears to be an Apple extension per:
// http://www.unix.com/man-page/OSX/3/posix_spawnattr_setbinpref_np/
#if !defined(__arm__)

  // Don't set the binpref if a shell was provided.  After all, that's only
  // going to affect what version of the shell
  // is launched, not what fork of the binary is launched.  We insert "arch
  // --arch <ARCH> as part of the shell invocation
  // to do that job on OSX.

  if (launch_info.GetShell() == nullptr) {
    // We don't need to do this for ARM, and we really shouldn't now that we
    // have multiple CPU subtypes and no posix_spawnattr call that allows us
    // to set which CPU subtype to launch...
    const ArchSpec &arch_spec = launch_info.GetArchitecture();
    cpu_type_t cpu = arch_spec.GetMachOCPUType();
    cpu_type_t sub = arch_spec.GetMachOCPUSubType();
    if (cpu != 0 && cpu != static_cast<cpu_type_t>(UINT32_MAX) &&
        cpu != static_cast<cpu_type_t>(LLDB_INVALID_CPUTYPE) &&
        !(cpu == 0x01000007 && sub == 8)) // If haswell is specified, don't try
                                          // to set the CPU type or we will fail
    {
      size_t ocount = 0;
      error.SetError(::posix_spawnattr_setbinpref_np(&attr, 1, &cpu, &ocount),
                     eErrorTypePOSIX);
      if (error.Fail())
        LLDB_LOG(log,
                 "error: {0}, ::posix_spawnattr_setbinpref_np ( &attr, 1, "
                 "cpu_type = {1:x}, count => {2} )",
                 error, cpu, ocount);

      if (error.Fail() || ocount != 1)
        return error;
    }
  }
#endif // !defined(__arm__)

  const char *tmp_argv[2];
  char *const *argv = const_cast<char *const *>(
      launch_info.GetArguments().GetConstArgumentVector());
  Environment::Envp envp = launch_info.GetEnvironment().getEnvp();
  if (argv == NULL) {
    // posix_spawn gets very unhappy if it doesn't have at least the program
    // name in argv[0]. One of the side affects I have noticed is the
    // environment
    // variables don't make it into the child process if "argv == NULL"!!!
    tmp_argv[0] = exe_path;
    tmp_argv[1] = NULL;
    argv = const_cast<char *const *>(tmp_argv);
  }

  FileSpec working_dir{launch_info.GetWorkingDirectory()};
  if (working_dir) {
    // Set the working directory on this thread only
    if (__pthread_chdir(working_dir.GetCString()) < 0) {
      if (errno == ENOENT) {
        error.SetErrorStringWithFormat("No such file or directory: %s",
                                       working_dir.GetCString());
      } else if (errno == ENOTDIR) {
        error.SetErrorStringWithFormat("Path doesn't name a directory: %s",
                                       working_dir.GetCString());
      } else {
        error.SetErrorStringWithFormat("An unknown error occurred when "
                                       "changing directory for process "
                                       "execution.");
      }
      return error;
    }
  }

  ::pid_t result_pid = LLDB_INVALID_PROCESS_ID;
  const size_t num_file_actions = launch_info.GetNumFileActions();
  if (num_file_actions > 0) {
    posix_spawn_file_actions_t file_actions;
    error.SetError(::posix_spawn_file_actions_init(&file_actions),
                   eErrorTypePOSIX);
    if (error.Fail()) {
      LLDB_LOG(log,
               "error: {0}, ::posix_spawn_file_actions_init ( &file_actions )",
               error);
      return error;
    }

    // Make sure we clean up the posix file actions before exiting this scope.
    CleanUp cleanup_fileact(posix_spawn_file_actions_destroy, &file_actions);

    for (size_t i = 0; i < num_file_actions; ++i) {
      const FileAction *launch_file_action =
          launch_info.GetFileActionAtIndex(i);
      if (launch_file_action) {
        if (!AddPosixSpawnFileAction(&file_actions, launch_file_action, log,
                                     error))
          return error;
      }
    }

    error.SetError(
        ::posix_spawnp(&result_pid, exe_path, &file_actions, &attr, argv, envp),
        eErrorTypePOSIX);

    if (error.Fail()) {
      LLDB_LOG(log,
               "error: {0}, ::posix_spawnp(pid => {1}, path = '{2}', "
               "file_actions = {3}, "
               "attr = {4}, argv = {5}, envp = {6} )",
               error, result_pid, exe_path, &file_actions, &attr, argv,
               envp.get());
      if (log) {
        for (int ii = 0; argv[ii]; ++ii)
          LLDB_LOG(log, "argv[{0}] = '{1}'", ii, argv[ii]);
      }
    }

  } else {
    error.SetError(
        ::posix_spawnp(&result_pid, exe_path, NULL, &attr, argv, envp),
        eErrorTypePOSIX);

    if (error.Fail()) {
      LLDB_LOG(log,
               "error: {0}, ::posix_spawnp ( pid => {1}, path = '{2}', "
               "file_actions = NULL, attr = {3}, argv = {4}, envp = {5} )",
               error, result_pid, exe_path, &attr, argv, envp.get());
      if (log) {
        for (int ii = 0; argv[ii]; ++ii)
          LLDB_LOG(log, "argv[{0}] = '{1}'", ii, argv[ii]);
      }
    }
  }
  pid = result_pid;

  if (working_dir) {
    // No more thread specific current working directory
    __pthread_fchdir(-1);
  }

  return error;
}

static bool ShouldLaunchUsingXPC(ProcessLaunchInfo &launch_info) {
  bool result = false;

#if !NO_XPC_SERVICES
  bool launchingAsRoot = launch_info.GetUserID() == 0;
  bool currentUserIsRoot = HostInfo::GetEffectiveUserID() == 0;

  if (launchingAsRoot && !currentUserIsRoot) {
    // If current user is already root, we don't need XPC's help.
    result = true;
  }
#endif

  return result;
}

Status Host::LaunchProcess(ProcessLaunchInfo &launch_info) {
  Status error;

  FileSystem &fs = FileSystem::Instance();
  FileSpec exe_spec(launch_info.GetExecutableFile());

  if (!fs.Exists(exe_spec))
    FileSystem::Instance().Resolve(exe_spec);

  if (!fs.Exists(exe_spec))
    FileSystem::Instance().ResolveExecutableLocation(exe_spec);

  if (!fs.Exists(exe_spec)) {
    error.SetErrorStringWithFormatv("executable doesn't exist: '{0}'",
                                    exe_spec);
    return error;
  }

  if (launch_info.GetFlags().Test(eLaunchFlagLaunchInTTY)) {
#if !defined(__arm__) && !defined(__arm64__) && !defined(__aarch64__)
    return LaunchInNewTerminalWithAppleScript(exe_spec.GetPath().c_str(),
                                              launch_info);
#else
    error.SetErrorString("launching a process in a new terminal is not "
                         "supported on iOS devices");
    return error;
#endif
  }

  lldb::pid_t pid = LLDB_INVALID_PROCESS_ID;

  // From now on we'll deal with the external (devirtualized) path.
  auto exe_path = fs.GetExternalPath(exe_spec);
  if (!exe_path)
    return Status(exe_path.getError());

  if (ShouldLaunchUsingXPC(launch_info))
    error = LaunchProcessXPC(exe_path->c_str(), launch_info, pid);
  else
    error = LaunchProcessPosixSpawn(exe_path->c_str(), launch_info, pid);

  if (pid != LLDB_INVALID_PROCESS_ID) {
    // If all went well, then set the process ID into the launch info
    launch_info.SetProcessID(pid);

    // Make sure we reap any processes we spawn or we will have zombies.
    bool monitoring = launch_info.MonitorProcess();
    UNUSED_IF_ASSERT_DISABLED(monitoring);
    assert(monitoring);
  } else {
    // Invalid process ID, something didn't go well
    if (error.Success())
      error.SetErrorString("process launch failed for unknown reasons");
  }
  return error;
}

Status Host::ShellExpandArguments(ProcessLaunchInfo &launch_info) {
  Status error;
  if (launch_info.GetFlags().Test(eLaunchFlagShellExpandArguments)) {
    FileSpec expand_tool_spec = HostInfo::GetSupportExeDir();
    if (!expand_tool_spec) {
      error.SetErrorString(
          "could not get support executable directory for lldb-argdumper tool");
      return error;
    }
    expand_tool_spec.AppendPathComponent("lldb-argdumper");
    if (!FileSystem::Instance().Exists(expand_tool_spec)) {
      error.SetErrorStringWithFormat(
          "could not find the lldb-argdumper tool: %s",
          expand_tool_spec.GetPath().c_str());
      return error;
    }

    StreamString expand_tool_spec_stream;
    expand_tool_spec_stream.Printf("\"%s\"",
                                   expand_tool_spec.GetPath().c_str());

    Args expand_command(expand_tool_spec_stream.GetData());
    expand_command.AppendArguments(launch_info.GetArguments());

    int status;
    std::string output;
    FileSpec cwd(launch_info.GetWorkingDirectory());
    if (!FileSystem::Instance().Exists(cwd)) {
      char *wd = getcwd(nullptr, 0);
      if (wd == nullptr) {
        error.SetErrorStringWithFormat(
            "cwd does not exist; cannot launch with shell argument expansion");
        return error;
      } else {
        FileSpec working_dir(wd);
        free(wd);
        launch_info.SetWorkingDirectory(working_dir);
      }
    }
    RunShellCommand(expand_command, cwd, &status, nullptr, &output,
                    std::chrono::seconds(10));

    if (status != 0) {
      error.SetErrorStringWithFormat("lldb-argdumper exited with error %d",
                                     status);
      return error;
    }

    auto data_sp = StructuredData::ParseJSON(output);
    if (!data_sp) {
      error.SetErrorString("invalid JSON");
      return error;
    }

    auto dict_sp = data_sp->GetAsDictionary();
    if (!data_sp) {
      error.SetErrorString("invalid JSON");
      return error;
    }

    auto args_sp = dict_sp->GetObjectForDotSeparatedPath("arguments");
    if (!args_sp) {
      error.SetErrorString("invalid JSON");
      return error;
    }

    auto args_array_sp = args_sp->GetAsArray();
    if (!args_array_sp) {
      error.SetErrorString("invalid JSON");
      return error;
    }

    launch_info.GetArguments().Clear();

    for (size_t i = 0; i < args_array_sp->GetSize(); i++) {
      auto item_sp = args_array_sp->GetItemAtIndex(i);
      if (!item_sp)
        continue;
      auto str_sp = item_sp->GetAsString();
      if (!str_sp)
        continue;

      launch_info.GetArguments().AppendArgument(str_sp->GetValue());
    }
  }

  return error;
}

HostThread Host::StartMonitoringChildProcess(
    const Host::MonitorChildProcessCallback &callback, lldb::pid_t pid,
    bool monitor_signals) {
  unsigned long mask = DISPATCH_PROC_EXIT;
  if (monitor_signals)
    mask |= DISPATCH_PROC_SIGNAL;

  Log *log(lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_HOST |
                                                  LIBLLDB_LOG_PROCESS));

  dispatch_source_t source = ::dispatch_source_create(
      DISPATCH_SOURCE_TYPE_PROC, pid, mask,
      ::dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0));

  if (log)
    log->Printf("Host::StartMonitoringChildProcess "
                "(callback, pid=%i, monitor_signals=%i) "
                "source = %p\n",
                static_cast<int>(pid), monitor_signals,
                reinterpret_cast<void *>(source));

  if (source) {
    Host::MonitorChildProcessCallback callback_copy = callback;
    ::dispatch_source_set_cancel_handler(source, ^{
      dispatch_release(source);
    });
    ::dispatch_source_set_event_handler(source, ^{

      int status = 0;
      int wait_pid = 0;
      bool cancel = false;
      bool exited = false;
      wait_pid = llvm::sys::RetryAfterSignal(-1, ::waitpid, pid, &status, 0);
      if (wait_pid >= 0) {
        int signal = 0;
        int exit_status = 0;
        const char *status_cstr = NULL;
        if (WIFSTOPPED(status)) {
          signal = WSTOPSIG(status);
          status_cstr = "STOPPED";
        } else if (WIFEXITED(status)) {
          exit_status = WEXITSTATUS(status);
          status_cstr = "EXITED";
          exited = true;
        } else if (WIFSIGNALED(status)) {
          signal = WTERMSIG(status);
          status_cstr = "SIGNALED";
          exited = true;
          exit_status = -1;
        } else {
          status_cstr = "???";
        }

        if (log)
          log->Printf("::waitpid (pid = %llu, &status, 0) => pid = %i, status "
                      "= 0x%8.8x (%s), signal = %i, exit_status = %i",
                      pid, wait_pid, status, status_cstr, signal, exit_status);

        if (callback_copy)
          cancel = callback_copy(pid, exited, signal, exit_status);

        if (exited || cancel) {
          ::dispatch_source_cancel(source);
        }
      }
    });

    ::dispatch_resume(source);
  }
  return HostThread();
}

//----------------------------------------------------------------------
// Log to both stderr and to ASL Logging when running on MacOSX.
//----------------------------------------------------------------------
void Host::SystemLog(SystemLogType type, const char *format, va_list args) {
  if (format && format[0]) {
    static aslmsg g_aslmsg = NULL;
    if (g_aslmsg == NULL) {
      g_aslmsg = ::asl_new(ASL_TYPE_MSG);
      char asl_key_sender[PATH_MAX];
      snprintf(asl_key_sender, sizeof(asl_key_sender),
               "com.apple.LLDB.framework");
      ::asl_set(g_aslmsg, ASL_KEY_SENDER, asl_key_sender);
    }

    // Copy the va_list so we can log this message twice
    va_list copy_args;
    va_copy(copy_args, args);
    // Log to stderr
    ::vfprintf(stderr, format, copy_args);
    va_end(copy_args);

    int asl_level;
    switch (type) {
    case eSystemLogError:
      asl_level = ASL_LEVEL_ERR;
      break;

    case eSystemLogWarning:
      asl_level = ASL_LEVEL_WARNING;
      break;
    }

    // Log to ASL
    ::asl_vlog(NULL, g_aslmsg, asl_level, format, args);
  }
}
