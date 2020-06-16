//===-- MachProcess.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 6/15/07.
//
//===----------------------------------------------------------------------===//

#include "DNB.h"
#include "MacOSX/CFUtils.h"
#include "SysSignal.h"
#include <dlfcn.h>
#include <inttypes.h>
#include <mach-o/loader.h>
#include <mach/mach.h>
#include <mach/task.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <sys/fcntl.h>
#include <sys/ptrace.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <uuid/uuid.h>

#include <algorithm>
#include <chrono>
#include <map>

#include <TargetConditionals.h>
#import <Foundation/Foundation.h>

#include "DNBDataRef.h"
#include "DNBLog.h"
#include "DNBThreadResumeActions.h"
#include "DNBTimer.h"
#include "MachProcess.h"
#include "PseudoTerminal.h"

#include "CFBundle.h"
#include "CFString.h"

#ifndef PLATFORM_BRIDGEOS
#define PLATFORM_BRIDGEOS 5
#endif

#ifndef PLATFORM_MACCATALYST
#define PLATFORM_MACCATALYST 6
#endif

#ifndef PLATFORM_IOSSIMULATOR
#define PLATFORM_IOSSIMULATOR 7
#endif

#ifndef PLATFORM_TVOSSIMULATOR
#define PLATFORM_TVOSSIMULATOR 8
#endif

#ifndef PLATFORM_WATCHOSSIMULATOR
#define PLATFORM_WATCHOSSIMULATOR 9
#endif

#ifndef PLATFORM_DRIVERKIT
#define PLATFORM_DRIVERKIT 10
#endif

#ifdef WITH_SPRINGBOARD

#include <CoreFoundation/CoreFoundation.h>
#include <SpringBoardServices/SBSWatchdogAssertion.h>
#include <SpringBoardServices/SpringBoardServer.h>

#endif // WITH_SPRINGBOARD

#if defined(WITH_SPRINGBOARD) || defined(WITH_BKS) || defined(WITH_FBS)
// This returns a CFRetained pointer to the Bundle ID for app_bundle_path,
// or NULL if there was some problem getting the bundle id.
static CFStringRef CopyBundleIDForPath(const char *app_bundle_path,
                                       DNBError &err_str);
#endif

#if defined(WITH_BKS) || defined(WITH_FBS)
#import <Foundation/Foundation.h>
static const int OPEN_APPLICATION_TIMEOUT_ERROR = 111;
typedef void (*SetErrorFunction)(NSInteger, std::string, DNBError &);
typedef bool (*CallOpenApplicationFunction)(NSString *bundleIDNSStr,
                                            NSDictionary *options,
                                            DNBError &error, pid_t *return_pid);
// This function runs the BKSSystemService (or FBSSystemService) method
// openApplication:options:clientPort:withResult,
// messaging the app passed in bundleIDNSStr.
// The function should be run inside of an NSAutoReleasePool.
//
// It will use the "options" dictionary passed in, and fill the error passed in
// if there is an error.
// If return_pid is not NULL, we'll fetch the pid that was made for the
// bundleID.
// If bundleIDNSStr is NULL, then the system application will be messaged.

template <typename OpenFlavor, typename ErrorFlavor,
          ErrorFlavor no_error_enum_value, SetErrorFunction error_function>
static bool CallBoardSystemServiceOpenApplication(NSString *bundleIDNSStr,
                                                  NSDictionary *options,
                                                  DNBError &error,
                                                  pid_t *return_pid) {
  // Now make our systemService:
  OpenFlavor *system_service = [[OpenFlavor alloc] init];

  if (bundleIDNSStr == nil) {
    bundleIDNSStr = [system_service systemApplicationBundleIdentifier];
    if (bundleIDNSStr == nil) {
      // Okay, no system app...
      error.SetErrorString("No system application to message.");
      return false;
    }
  }

  mach_port_t client_port = [system_service createClientPort];
  __block dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
  __block ErrorFlavor open_app_error = no_error_enum_value;
  __block std::string open_app_error_string;
  bool wants_pid = (return_pid != NULL);
  __block pid_t pid_in_block;

  const char *cstr = [bundleIDNSStr UTF8String];
  if (!cstr)
    cstr = "<Unknown Bundle ID>";

  NSString *description = [options description];
  DNBLog("About to launch process for bundle ID: %s - options:\n%s", cstr,
    [description UTF8String]);
  [system_service
      openApplication:bundleIDNSStr
              options:options
           clientPort:client_port
           withResult:^(NSError *bks_error) {
             // The system service will cleanup the client port we created for
             // us.
             if (bks_error)
               open_app_error = (ErrorFlavor)[bks_error code];

             if (open_app_error == no_error_enum_value) {
               if (wants_pid) {
                 pid_in_block =
                     [system_service pidForApplication:bundleIDNSStr];
                 DNBLog(
                     "In completion handler, got pid for bundle id, pid: %d.",
                     pid_in_block);
                 DNBLogThreadedIf(
                     LOG_PROCESS,
                     "In completion handler, got pid for bundle id, pid: %d.",
                     pid_in_block);
               } else
                 DNBLogThreadedIf(LOG_PROCESS,
                                  "In completion handler: success.");
             } else {
               const char *error_str =
                   [(NSString *)[bks_error localizedDescription] UTF8String];
               if (error_str) {
                 open_app_error_string = error_str;
                 DNBLogError("In app launch attempt, got error "
                             "localizedDescription '%s'.", error_str);
                 const char *obj_desc = 
                      [NSString stringWithFormat:@"%@", bks_error].UTF8String;
                 DNBLogError("In app launch attempt, got error "
                             "NSError object description: '%s'.",
                             obj_desc);
               }
               DNBLogThreadedIf(LOG_PROCESS, "In completion handler for send "
                                             "event, got error \"%s\"(%ld).",
                                error_str ? error_str : "<unknown error>",
                                open_app_error);
             }

             [system_service release];
             dispatch_semaphore_signal(semaphore);
           }

  ];

  const uint32_t timeout_secs = 30;

  dispatch_time_t timeout =
      dispatch_time(DISPATCH_TIME_NOW, timeout_secs * NSEC_PER_SEC);

  long success = dispatch_semaphore_wait(semaphore, timeout) == 0;

  dispatch_release(semaphore);

  if (!success) {
    DNBLogError("timed out trying to send openApplication to %s.", cstr);
    error.SetError(OPEN_APPLICATION_TIMEOUT_ERROR, DNBError::Generic);
    error.SetErrorString("timed out trying to launch app");
  } else if (open_app_error != no_error_enum_value) {
    error_function(open_app_error, open_app_error_string, error);
    DNBLogError("unable to launch the application with CFBundleIdentifier '%s' "
                "bks_error = %u",
                cstr, open_app_error);
    success = false;
  } else if (wants_pid) {
    *return_pid = pid_in_block;
    DNBLogThreadedIf(
        LOG_PROCESS,
        "Out of completion handler, pid from block %d and passing out: %d",
        pid_in_block, *return_pid);
  }

  return success;
}
#endif

#if defined(WITH_BKS) || defined(WITH_FBS)
static void SplitEventData(const char *data, std::vector<std::string> &elements)
{
  elements.clear();
  if (!data)
    return;

  const char *start = data;

  while (*start != '\0') {
    const char *token = strchr(start, ':');
    if (!token) {
      elements.push_back(std::string(start));
      return;
    }
    if (token != start)
      elements.push_back(std::string(start, token - start));
    start = ++token;
  }
}
#endif

#ifdef WITH_BKS
#import <Foundation/Foundation.h>
extern "C" {
#import <BackBoardServices/BKSOpenApplicationConstants_Private.h>
#import <BackBoardServices/BKSSystemService_LaunchServices.h>
#import <BackBoardServices/BackBoardServices.h>
}

static bool IsBKSProcess(nub_process_t pid) {
  BKSApplicationStateMonitor *state_monitor =
      [[BKSApplicationStateMonitor alloc] init];
  BKSApplicationState app_state =
      [state_monitor mostElevatedApplicationStateForPID:pid];
  return app_state != BKSApplicationStateUnknown;
}

static void SetBKSError(NSInteger error_code, 
                        std::string error_description, 
                        DNBError &error) {
  error.SetError(error_code, DNBError::BackBoard);
  NSString *err_nsstr = ::BKSOpenApplicationErrorCodeToString(
      (BKSOpenApplicationErrorCode)error_code);
  std::string err_str = "unknown BKS error";
  if (error_description.empty() == false) {
    err_str = error_description;
  } else if (err_nsstr != nullptr) {
    err_str = [err_nsstr UTF8String];
  }
  error.SetErrorString(err_str.c_str());
}

static bool BKSAddEventDataToOptions(NSMutableDictionary *options,
                                     const char *event_data,
                                     DNBError &option_error) {
  std::vector<std::string> values;
  SplitEventData(event_data, values);
  bool found_one = false;
  for (std::string value : values)
  {
      if (value.compare("BackgroundContentFetching") == 0) {
        DNBLog("Setting ActivateForEvent key in options dictionary.");
        NSDictionary *event_details = [NSDictionary dictionary];
        NSDictionary *event_dictionary = [NSDictionary
            dictionaryWithObject:event_details
                          forKey:
                              BKSActivateForEventOptionTypeBackgroundContentFetching];
        [options setObject:event_dictionary
                    forKey:BKSOpenApplicationOptionKeyActivateForEvent];
        found_one = true;
      } else if (value.compare("ActivateSuspended") == 0) {
        DNBLog("Setting ActivateSuspended key in options dictionary.");
        [options setObject:@YES forKey: BKSOpenApplicationOptionKeyActivateSuspended];
        found_one = true;
      } else {
        DNBLogError("Unrecognized event type: %s.  Ignoring.", value.c_str());
        option_error.SetErrorString("Unrecognized event data");
      }
  }
  return found_one;
}

static NSMutableDictionary *BKSCreateOptionsDictionary(
    const char *app_bundle_path, NSMutableArray *launch_argv,
    NSMutableDictionary *launch_envp, NSString *stdio_path, bool disable_aslr,
    const char *event_data) {
  NSMutableDictionary *debug_options = [NSMutableDictionary dictionary];
  if (launch_argv != nil)
    [debug_options setObject:launch_argv forKey:BKSDebugOptionKeyArguments];
  if (launch_envp != nil)
    [debug_options setObject:launch_envp forKey:BKSDebugOptionKeyEnvironment];

  [debug_options setObject:stdio_path forKey:BKSDebugOptionKeyStandardOutPath];
  [debug_options setObject:stdio_path
                    forKey:BKSDebugOptionKeyStandardErrorPath];
  [debug_options setObject:[NSNumber numberWithBool:YES]
                    forKey:BKSDebugOptionKeyWaitForDebugger];
  if (disable_aslr)
    [debug_options setObject:[NSNumber numberWithBool:YES]
                      forKey:BKSDebugOptionKeyDisableASLR];

  // That will go in the overall dictionary:

  NSMutableDictionary *options = [NSMutableDictionary dictionary];
  [options setObject:debug_options
              forKey:BKSOpenApplicationOptionKeyDebuggingOptions];
  // And there are some other options at the top level in this dictionary:
  [options setObject:[NSNumber numberWithBool:YES]
              forKey:BKSOpenApplicationOptionKeyUnlockDevice];

  DNBError error;
  BKSAddEventDataToOptions(options, event_data, error);

  return options;
}

static CallOpenApplicationFunction BKSCallOpenApplicationFunction =
    CallBoardSystemServiceOpenApplication<
        BKSSystemService, BKSOpenApplicationErrorCode,
        BKSOpenApplicationErrorCodeNone, SetBKSError>;
#endif // WITH_BKS

#ifdef WITH_FBS
#import <Foundation/Foundation.h>
extern "C" {
#import <FrontBoardServices/FBSOpenApplicationConstants_Private.h>
#import <FrontBoardServices/FBSSystemService_LaunchServices.h>
#import <FrontBoardServices/FrontBoardServices.h>
#import <MobileCoreServices/LSResourceProxy.h>
#import <MobileCoreServices/MobileCoreServices.h>
}

#ifdef WITH_BKS
static bool IsFBSProcess(nub_process_t pid) {
  BKSApplicationStateMonitor *state_monitor =
      [[BKSApplicationStateMonitor alloc] init];
  BKSApplicationState app_state =
      [state_monitor mostElevatedApplicationStateForPID:pid];
  return app_state != BKSApplicationStateUnknown;
}
#else
static bool IsFBSProcess(nub_process_t pid) {
  // FIXME: What is the FBS equivalent of BKSApplicationStateMonitor
  return false;
}
#endif

static void SetFBSError(NSInteger error_code, 
                        std::string error_description, 
                        DNBError &error) {
  error.SetError((DNBError::ValueType)error_code, DNBError::FrontBoard);
  NSString *err_nsstr = ::FBSOpenApplicationErrorCodeToString(
      (FBSOpenApplicationErrorCode)error_code);
  std::string err_str = "unknown FBS error";
  if (error_description.empty() == false) {
    err_str = error_description;
  } else if (err_nsstr != nullptr) {
    err_str = [err_nsstr UTF8String];
  }
  error.SetErrorString(err_str.c_str());
}

static bool FBSAddEventDataToOptions(NSMutableDictionary *options,
                                     const char *event_data,
                                     DNBError &option_error) {
  std::vector<std::string> values;
  SplitEventData(event_data, values);
  bool found_one = false;
  for (std::string value : values)
  {
      if (value.compare("BackgroundContentFetching") == 0) {
        DNBLog("Setting ActivateForEvent key in options dictionary.");
        NSDictionary *event_details = [NSDictionary dictionary];
        NSDictionary *event_dictionary = [NSDictionary
            dictionaryWithObject:event_details
                          forKey:
                              FBSActivateForEventOptionTypeBackgroundContentFetching];
        [options setObject:event_dictionary
                    forKey:FBSOpenApplicationOptionKeyActivateForEvent];
        found_one = true;
      } else if (value.compare("ActivateSuspended") == 0) {
        DNBLog("Setting ActivateSuspended key in options dictionary.");
        [options setObject:@YES forKey: FBSOpenApplicationOptionKeyActivateSuspended];
        found_one = true;
      } else {
        DNBLogError("Unrecognized event type: %s.  Ignoring.", value.c_str());
        option_error.SetErrorString("Unrecognized event data.");
      }
  }
  return found_one;
}

static NSMutableDictionary *
FBSCreateOptionsDictionary(const char *app_bundle_path,
                           NSMutableArray *launch_argv,
                           NSDictionary *launch_envp, NSString *stdio_path,
                           bool disable_aslr, const char *event_data) {
  NSMutableDictionary *debug_options = [NSMutableDictionary dictionary];

  if (launch_argv != nil)
    [debug_options setObject:launch_argv forKey:FBSDebugOptionKeyArguments];
  if (launch_envp != nil)
    [debug_options setObject:launch_envp forKey:FBSDebugOptionKeyEnvironment];

  [debug_options setObject:stdio_path forKey:FBSDebugOptionKeyStandardOutPath];
  [debug_options setObject:stdio_path
                    forKey:FBSDebugOptionKeyStandardErrorPath];
  [debug_options setObject:[NSNumber numberWithBool:YES]
                    forKey:FBSDebugOptionKeyWaitForDebugger];
  if (disable_aslr)
    [debug_options setObject:[NSNumber numberWithBool:YES]
                      forKey:FBSDebugOptionKeyDisableASLR];

  // That will go in the overall dictionary:

  NSMutableDictionary *options = [NSMutableDictionary dictionary];
  [options setObject:debug_options
              forKey:FBSOpenApplicationOptionKeyDebuggingOptions];
  // And there are some other options at the top level in this dictionary:
  [options setObject:[NSNumber numberWithBool:YES]
              forKey:FBSOpenApplicationOptionKeyUnlockDevice];

  // We have to get the "sequence ID & UUID" for this app bundle path and send
  // them to FBS:

  NSURL *app_bundle_url =
      [NSURL fileURLWithPath:[NSString stringWithUTF8String:app_bundle_path]
                 isDirectory:YES];
  LSApplicationProxy *app_proxy =
      [LSApplicationProxy applicationProxyForBundleURL:app_bundle_url];
  if (app_proxy) {
    DNBLog("Sending AppProxy info: sequence no: %lu, GUID: %s.",
           app_proxy.sequenceNumber,
           [app_proxy.cacheGUID.UUIDString UTF8String]);
    [options
        setObject:[NSNumber numberWithUnsignedInteger:app_proxy.sequenceNumber]
           forKey:FBSOpenApplicationOptionKeyLSSequenceNumber];
    [options setObject:app_proxy.cacheGUID.UUIDString
                forKey:FBSOpenApplicationOptionKeyLSCacheGUID];
  }

  DNBError error;
  FBSAddEventDataToOptions(options, event_data, error);

  return options;
}
static CallOpenApplicationFunction FBSCallOpenApplicationFunction =
    CallBoardSystemServiceOpenApplication<
        FBSSystemService, FBSOpenApplicationErrorCode,
        FBSOpenApplicationErrorCodeNone, SetFBSError>;
#endif // WITH_FBS

#if 0
#define DEBUG_LOG(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#define DEBUG_LOG(fmt, ...)
#endif

#ifndef MACH_PROCESS_USE_POSIX_SPAWN
#define MACH_PROCESS_USE_POSIX_SPAWN 1
#endif

#ifndef _POSIX_SPAWN_DISABLE_ASLR
#define _POSIX_SPAWN_DISABLE_ASLR 0x0100
#endif

MachProcess::MachProcess()
    : m_pid(0), m_cpu_type(0), m_child_stdin(-1), m_child_stdout(-1),
      m_child_stderr(-1), m_path(), m_args(), m_task(this),
      m_flags(eMachProcessFlagsNone), m_stdio_thread(0),
      m_stdio_mutex(PTHREAD_MUTEX_RECURSIVE), m_stdout_data(),
      m_profile_enabled(false), m_profile_interval_usec(0), m_profile_thread(0),
      m_profile_data_mutex(PTHREAD_MUTEX_RECURSIVE), m_profile_data(),
      m_profile_events(0, eMachProcessProfileCancel),
      m_thread_actions(), m_exception_messages(),
      m_exception_messages_mutex(PTHREAD_MUTEX_RECURSIVE), m_thread_list(),
      m_activities(), m_state(eStateUnloaded),
      m_state_mutex(PTHREAD_MUTEX_RECURSIVE), m_events(0, kAllEventsMask),
      m_private_events(0, kAllEventsMask), m_breakpoints(), m_watchpoints(),
      m_name_to_addr_callback(NULL), m_name_to_addr_baton(NULL),
      m_image_infos_callback(NULL), m_image_infos_baton(NULL),
      m_sent_interrupt_signo(0), m_auto_resume_signo(0), m_did_exec(false),
      m_dyld_process_info_create(nullptr),
      m_dyld_process_info_for_each_image(nullptr),
      m_dyld_process_info_release(nullptr),
      m_dyld_process_info_get_cache(nullptr) {
  m_dyld_process_info_create =
      (void *(*)(task_t task, uint64_t timestamp, kern_return_t * kernelError))
          dlsym(RTLD_DEFAULT, "_dyld_process_info_create");
  m_dyld_process_info_for_each_image =
      (void (*)(void *info, void (^)(uint64_t machHeaderAddress,
                                     const uuid_t uuid, const char *path)))
          dlsym(RTLD_DEFAULT, "_dyld_process_info_for_each_image");
  m_dyld_process_info_release =
      (void (*)(void *info))dlsym(RTLD_DEFAULT, "_dyld_process_info_release");
  m_dyld_process_info_get_cache = (void (*)(void *info, void *cacheInfo))dlsym(
      RTLD_DEFAULT, "_dyld_process_info_get_cache");
  m_dyld_process_info_get_platform = (uint32_t (*)(void *info))dlsym(
      RTLD_DEFAULT, "_dyld_process_info_get_platform");

  DNBLogThreadedIf(LOG_PROCESS | LOG_VERBOSE, "%s", __PRETTY_FUNCTION__);
}

MachProcess::~MachProcess() {
  DNBLogThreadedIf(LOG_PROCESS | LOG_VERBOSE, "%s", __PRETTY_FUNCTION__);
  Clear();
}

pid_t MachProcess::SetProcessID(pid_t pid) {
  // Free any previous process specific data or resources
  Clear();
  // Set the current PID appropriately
  if (pid == 0)
    m_pid = ::getpid();
  else
    m_pid = pid;
  return m_pid; // Return actually PID in case a zero pid was passed in
}

nub_state_t MachProcess::GetState() {
  // If any other threads access this we will need a mutex for it
  PTHREAD_MUTEX_LOCKER(locker, m_state_mutex);
  return m_state;
}

const char *MachProcess::ThreadGetName(nub_thread_t tid) {
  return m_thread_list.GetName(tid);
}

nub_state_t MachProcess::ThreadGetState(nub_thread_t tid) {
  return m_thread_list.GetState(tid);
}

nub_size_t MachProcess::GetNumThreads() const {
  return m_thread_list.NumThreads();
}

nub_thread_t MachProcess::GetThreadAtIndex(nub_size_t thread_idx) const {
  return m_thread_list.ThreadIDAtIndex(thread_idx);
}

nub_thread_t
MachProcess::GetThreadIDForMachPortNumber(thread_t mach_port_number) const {
  return m_thread_list.GetThreadIDByMachPortNumber(mach_port_number);
}

nub_bool_t MachProcess::SyncThreadState(nub_thread_t tid) {
  MachThreadSP thread_sp(m_thread_list.GetThreadByID(tid));
  if (!thread_sp)
    return false;
  kern_return_t kret = ::thread_abort_safely(thread_sp->MachPortNumber());
  DNBLogThreadedIf(LOG_THREAD, "thread = 0x%8.8" PRIx32
                               " calling thread_abort_safely (tid) => %u "
                               "(GetGPRState() for stop_count = %u)",
                   thread_sp->MachPortNumber(), kret,
                   thread_sp->Process()->StopCount());

  if (kret == KERN_SUCCESS)
    return true;
  else
    return false;
}

ThreadInfo::QoS MachProcess::GetRequestedQoS(nub_thread_t tid, nub_addr_t tsd,
                                             uint64_t dti_qos_class_index) {
  return m_thread_list.GetRequestedQoS(tid, tsd, dti_qos_class_index);
}

nub_addr_t MachProcess::GetPThreadT(nub_thread_t tid) {
  return m_thread_list.GetPThreadT(tid);
}

nub_addr_t MachProcess::GetDispatchQueueT(nub_thread_t tid) {
  return m_thread_list.GetDispatchQueueT(tid);
}

nub_addr_t MachProcess::GetTSDAddressForThread(
    nub_thread_t tid, uint64_t plo_pthread_tsd_base_address_offset,
    uint64_t plo_pthread_tsd_base_offset, uint64_t plo_pthread_tsd_entry_size) {
  return m_thread_list.GetTSDAddressForThread(
      tid, plo_pthread_tsd_base_address_offset, plo_pthread_tsd_base_offset,
      plo_pthread_tsd_entry_size);
}

/// Determine whether this is running on macOS.
/// Since debugserver runs on the same machine as the process, we can
/// just look at the compilation target.
static bool IsMacOSHost() {
#if TARGET_OS_OSX == 1
  return true;
#else
  return false;
#endif
}

const char *MachProcess::GetDeploymentInfo(const struct load_command& lc,
                                           uint64_t load_command_address,
                                           uint32_t& major_version,
                                           uint32_t& minor_version,
                                           uint32_t& patch_version) {
  uint32_t cmd = lc.cmd & ~LC_REQ_DYLD;
  bool lc_cmd_known =
    cmd == LC_VERSION_MIN_IPHONEOS || cmd == LC_VERSION_MIN_MACOSX ||
    cmd == LC_VERSION_MIN_TVOS || cmd == LC_VERSION_MIN_WATCHOS;

  if (lc_cmd_known) {
    struct version_min_command vers_cmd;
    if (ReadMemory(load_command_address, sizeof(struct version_min_command),
                   &vers_cmd) != sizeof(struct version_min_command)) {
      return nullptr;
    }
    major_version = vers_cmd.sdk >> 16;
    minor_version = (vers_cmd.sdk >> 8) & 0xffu;
    patch_version = vers_cmd.sdk & 0xffu;

    // Handle the older LC_VERSION load commands, which don't
    // distinguish between simulator and real hardware.
    switch (cmd) {
    case LC_VERSION_MIN_IPHONEOS:
      return IsMacOSHost() ? "iossimulator": "ios";
    case LC_VERSION_MIN_MACOSX:
      return "macosx";
    case LC_VERSION_MIN_TVOS:
      return IsMacOSHost() ? "tvossimulator": "tvos";
    case LC_VERSION_MIN_WATCHOS:
      return IsMacOSHost() ? "watchossimulator" : "watchos";
    default:
      return nullptr;
    }
  }
#if defined (LC_BUILD_VERSION)
  if (cmd == LC_BUILD_VERSION) {
    struct build_version_command build_vers;
    if (ReadMemory(load_command_address, sizeof(struct build_version_command),
                   &build_vers) != sizeof(struct build_version_command)) {
      return nullptr;
    }
    major_version = build_vers.sdk >> 16;;
    minor_version = (build_vers.sdk >> 8) & 0xffu;
    patch_version = build_vers.sdk & 0xffu;

    switch (build_vers.platform) {
    case PLATFORM_MACOS:
      return "macosx";
    case PLATFORM_MACCATALYST:
      return "maccatalyst";
    case PLATFORM_IOS:
      return "ios";
    case PLATFORM_IOSSIMULATOR:
      return "iossimulator";
    case PLATFORM_TVOS:
      return "tvos";
    case PLATFORM_TVOSSIMULATOR:
      return "tvossimulator";
    case PLATFORM_WATCHOS:
      return "watchos";
    case PLATFORM_WATCHOSSIMULATOR:
      return "watchossimulator";
    case PLATFORM_BRIDGEOS:
      return "bridgeos";
    case PLATFORM_DRIVERKIT:
      return "driverkit";
    }
  }
#endif
  return nullptr;
}

// Given an address, read the mach-o header and load commands out of memory to
// fill in
// the mach_o_information "inf" object.
//
// Returns false if there was an error in reading this mach-o file header/load
// commands.

bool MachProcess::GetMachOInformationFromMemory(
    uint32_t dyld_platform, nub_addr_t mach_o_header_addr, int wordsize,
    struct mach_o_information &inf) {
  uint64_t load_cmds_p;
  if (wordsize == 4) {
    struct mach_header header;
    if (ReadMemory(mach_o_header_addr, sizeof(struct mach_header), &header) !=
        sizeof(struct mach_header)) {
      return false;
    }
    load_cmds_p = mach_o_header_addr + sizeof(struct mach_header);
    inf.mach_header.magic = header.magic;
    inf.mach_header.cputype = header.cputype;
    // high byte of cpusubtype is used for "capability bits", v.
    // CPU_SUBTYPE_MASK, CPU_SUBTYPE_LIB64 in machine.h
    inf.mach_header.cpusubtype = header.cpusubtype & 0x00ffffff;
    inf.mach_header.filetype = header.filetype;
    inf.mach_header.ncmds = header.ncmds;
    inf.mach_header.sizeofcmds = header.sizeofcmds;
    inf.mach_header.flags = header.flags;
  } else {
    struct mach_header_64 header;
    if (ReadMemory(mach_o_header_addr, sizeof(struct mach_header_64),
                   &header) != sizeof(struct mach_header_64)) {
      return false;
    }
    load_cmds_p = mach_o_header_addr + sizeof(struct mach_header_64);
    inf.mach_header.magic = header.magic;
    inf.mach_header.cputype = header.cputype;
    // high byte of cpusubtype is used for "capability bits", v.
    // CPU_SUBTYPE_MASK, CPU_SUBTYPE_LIB64 in machine.h
    inf.mach_header.cpusubtype = header.cpusubtype & 0x00ffffff;
    inf.mach_header.filetype = header.filetype;
    inf.mach_header.ncmds = header.ncmds;
    inf.mach_header.sizeofcmds = header.sizeofcmds;
    inf.mach_header.flags = header.flags;
  }
  for (uint32_t j = 0; j < inf.mach_header.ncmds; j++) {
    struct load_command lc;
    if (ReadMemory(load_cmds_p, sizeof(struct load_command), &lc) !=
        sizeof(struct load_command)) {
      return false;
    }
    if (lc.cmd == LC_SEGMENT) {
      struct segment_command seg;
      if (ReadMemory(load_cmds_p, sizeof(struct segment_command), &seg) !=
          sizeof(struct segment_command)) {
        return false;
      }
      struct mach_o_segment this_seg;
      char name[17];
      ::memset(name, 0, sizeof(name));
      memcpy(name, seg.segname, sizeof(seg.segname));
      this_seg.name = name;
      this_seg.vmaddr = seg.vmaddr;
      this_seg.vmsize = seg.vmsize;
      this_seg.fileoff = seg.fileoff;
      this_seg.filesize = seg.filesize;
      this_seg.maxprot = seg.maxprot;
      this_seg.initprot = seg.initprot;
      this_seg.nsects = seg.nsects;
      this_seg.flags = seg.flags;
      inf.segments.push_back(this_seg);
      if (this_seg.name == "ExecExtraSuspend")
        m_task.TaskWillExecProcessesSuspended();
    }
    if (lc.cmd == LC_SEGMENT_64) {
      struct segment_command_64 seg;
      if (ReadMemory(load_cmds_p, sizeof(struct segment_command_64), &seg) !=
          sizeof(struct segment_command_64)) {
        return false;
      }
      struct mach_o_segment this_seg;
      char name[17];
      ::memset(name, 0, sizeof(name));
      memcpy(name, seg.segname, sizeof(seg.segname));
      this_seg.name = name;
      this_seg.vmaddr = seg.vmaddr;
      this_seg.vmsize = seg.vmsize;
      this_seg.fileoff = seg.fileoff;
      this_seg.filesize = seg.filesize;
      this_seg.maxprot = seg.maxprot;
      this_seg.initprot = seg.initprot;
      this_seg.nsects = seg.nsects;
      this_seg.flags = seg.flags;
      inf.segments.push_back(this_seg);
      if (this_seg.name == "ExecExtraSuspend")
        m_task.TaskWillExecProcessesSuspended();
    }
    if (lc.cmd == LC_UUID) {
      struct uuid_command uuidcmd;
      if (ReadMemory(load_cmds_p, sizeof(struct uuid_command), &uuidcmd) ==
          sizeof(struct uuid_command))
        uuid_copy(inf.uuid, uuidcmd.uuid);
    }

    uint32_t major_version, minor_version, patch_version;
    if (const char *lc_platform = GetDeploymentInfo(
            lc, load_cmds_p, major_version, minor_version, patch_version)) {
      // macCatalyst support.
      //
      // This handles two special cases:
      //
      // 1. Frameworks that have both a PLATFORM_MACOS and a
      //    PLATFORM_MACCATALYST load command.  Make sure to select
      //    the requested one.
      //
      // 2. The xctest binary is a pure macOS binary but is launched
      //    with DYLD_FORCE_PLATFORM=6.
      if (dyld_platform == PLATFORM_MACCATALYST &&
          inf.mach_header.filetype == MH_EXECUTE &&
          inf.min_version_os_name.empty() &&
          (strcmp("macosx", lc_platform) == 0)) {
        // DYLD says this *is* a macCatalyst process. If we haven't
        // parsed any load commands, transform a macOS load command
        // into a generic macCatalyst load command. It will be
        // overwritten by a more specific one if there is one.  This
        // is only done for the main executable. It is perfectly fine
        // for a macCatalyst binary to link against a macOS-only framework.
        inf.min_version_os_name = "maccatalyst";
        inf.min_version_os_version = GetMacCatalystVersionString();
      } else if (dyld_platform != PLATFORM_MACCATALYST &&
                 inf.min_version_os_name == "macosx") {
        // This is a binary with both PLATFORM_MACOS and
        // PLATFORM_MACCATALYST load commands and the process is not
        // running as PLATFORM_MACCATALYST. Stick with the
        // "macosx" load command that we've already processed,
        // ignore this one, which is presumed to be a
        // PLATFORM_MACCATALYST one.
      } else {
        inf.min_version_os_name = lc_platform;
        inf.min_version_os_version = "";
        inf.min_version_os_version += std::to_string(major_version);
        inf.min_version_os_version += ".";
        inf.min_version_os_version += std::to_string(minor_version);
        if (patch_version != 0) {
          inf.min_version_os_version += ".";
          inf.min_version_os_version += std::to_string(patch_version);
        }
      }
    }

    load_cmds_p += lc.cmdsize;
  }
  return true;
}

// Given completely filled in array of binary_image_information structures,
// create a JSONGenerator object
// with all the details we want to send to lldb.
JSONGenerator::ObjectSP MachProcess::FormatDynamicLibrariesIntoJSON(
    const std::vector<struct binary_image_information> &image_infos) {

  JSONGenerator::ArraySP image_infos_array_sp(new JSONGenerator::Array());

  const size_t image_count = image_infos.size();

  for (size_t i = 0; i < image_count; i++) {
    JSONGenerator::DictionarySP image_info_dict_sp(
        new JSONGenerator::Dictionary());
    image_info_dict_sp->AddIntegerItem("load_address",
                                       image_infos[i].load_address);
    image_info_dict_sp->AddIntegerItem("mod_date", image_infos[i].mod_date);
    image_info_dict_sp->AddStringItem("pathname", image_infos[i].filename);

    uuid_string_t uuidstr;
    uuid_unparse_upper(image_infos[i].macho_info.uuid, uuidstr);
    image_info_dict_sp->AddStringItem("uuid", uuidstr);

    if (!image_infos[i].macho_info.min_version_os_name.empty() &&
        !image_infos[i].macho_info.min_version_os_version.empty()) {
      image_info_dict_sp->AddStringItem(
          "min_version_os_name", image_infos[i].macho_info.min_version_os_name);
      image_info_dict_sp->AddStringItem(
          "min_version_os_sdk",
          image_infos[i].macho_info.min_version_os_version);
    }

    JSONGenerator::DictionarySP mach_header_dict_sp(
        new JSONGenerator::Dictionary());
    mach_header_dict_sp->AddIntegerItem(
        "magic", image_infos[i].macho_info.mach_header.magic);
    mach_header_dict_sp->AddIntegerItem(
        "cputype", (uint32_t)image_infos[i].macho_info.mach_header.cputype);
    mach_header_dict_sp->AddIntegerItem(
        "cpusubtype",
        (uint32_t)image_infos[i].macho_info.mach_header.cpusubtype);
    mach_header_dict_sp->AddIntegerItem(
        "filetype", image_infos[i].macho_info.mach_header.filetype);
    mach_header_dict_sp->AddIntegerItem ("flags", 
                         image_infos[i].macho_info.mach_header.flags);

    //          DynamicLoaderMacOSX doesn't currently need these fields, so
    //          don't send them.
    //            mach_header_dict_sp->AddIntegerItem ("ncmds",
    //            image_infos[i].macho_info.mach_header.ncmds);
    //            mach_header_dict_sp->AddIntegerItem ("sizeofcmds",
    //            image_infos[i].macho_info.mach_header.sizeofcmds);
    image_info_dict_sp->AddItem("mach_header", mach_header_dict_sp);

    JSONGenerator::ArraySP segments_sp(new JSONGenerator::Array());
    for (size_t j = 0; j < image_infos[i].macho_info.segments.size(); j++) {
      JSONGenerator::DictionarySP segment_sp(new JSONGenerator::Dictionary());
      segment_sp->AddStringItem("name",
                                image_infos[i].macho_info.segments[j].name);
      segment_sp->AddIntegerItem("vmaddr",
                                 image_infos[i].macho_info.segments[j].vmaddr);
      segment_sp->AddIntegerItem("vmsize",
                                 image_infos[i].macho_info.segments[j].vmsize);
      segment_sp->AddIntegerItem("fileoff",
                                 image_infos[i].macho_info.segments[j].fileoff);
      segment_sp->AddIntegerItem(
          "filesize", image_infos[i].macho_info.segments[j].filesize);
      segment_sp->AddIntegerItem("maxprot",
                                 image_infos[i].macho_info.segments[j].maxprot);

      //              DynamicLoaderMacOSX doesn't currently need these fields,
      //              so don't send them.
      //                segment_sp->AddIntegerItem ("initprot",
      //                image_infos[i].macho_info.segments[j].initprot);
      //                segment_sp->AddIntegerItem ("nsects",
      //                image_infos[i].macho_info.segments[j].nsects);
      //                segment_sp->AddIntegerItem ("flags",
      //                image_infos[i].macho_info.segments[j].flags);
      segments_sp->AddItem(segment_sp);
    }
    image_info_dict_sp->AddItem("segments", segments_sp);

    image_infos_array_sp->AddItem(image_info_dict_sp);
  }

  JSONGenerator::DictionarySP reply_sp(new JSONGenerator::Dictionary());
  ;
  reply_sp->AddItem("images", image_infos_array_sp);

  return reply_sp;
}

// Get the shared library information using the old (pre-macOS 10.12, pre-iOS
// 10, pre-tvOS 10, pre-watchOS 3)
// code path.  We'll be given the address of an array of structures in the form
// {void* load_addr, void* mod_date, void* pathname}
//
// In macOS 10.12 etc and newer, we'll use SPI calls into dyld to gather this
// information.
JSONGenerator::ObjectSP MachProcess::GetLoadedDynamicLibrariesInfos(
    nub_process_t pid, nub_addr_t image_list_address, nub_addr_t image_count) {
  JSONGenerator::DictionarySP reply_sp;

  int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PID, pid};
  struct kinfo_proc processInfo;
  size_t bufsize = sizeof(processInfo);
  if (sysctl(mib, (unsigned)(sizeof(mib) / sizeof(int)), &processInfo, &bufsize,
             NULL, 0) == 0 &&
      bufsize > 0) {
    uint32_t pointer_size = 4;
    if (processInfo.kp_proc.p_flag & P_LP64)
      pointer_size = 8;

    std::vector<struct binary_image_information> image_infos;
    size_t image_infos_size = image_count * 3 * pointer_size;

    uint8_t *image_info_buf = (uint8_t *)malloc(image_infos_size);
    if (image_info_buf == NULL) {
      return reply_sp;
    }
    if (ReadMemory(image_list_address, image_infos_size, image_info_buf) !=
        image_infos_size) {
      return reply_sp;
    }

    ////  First the image_infos array with (load addr, pathname, mod date)
    ///tuples

    for (size_t i = 0; i < image_count; i++) {
      struct binary_image_information info;
      nub_addr_t pathname_address;
      if (pointer_size == 4) {
        uint32_t load_address_32;
        uint32_t pathname_address_32;
        uint32_t mod_date_32;
        ::memcpy(&load_address_32, image_info_buf + (i * 3 * pointer_size), 4);
        ::memcpy(&pathname_address_32,
                 image_info_buf + (i * 3 * pointer_size) + pointer_size, 4);
        ::memcpy(&mod_date_32, image_info_buf + (i * 3 * pointer_size) +
                                   pointer_size + pointer_size,
                 4);
        info.load_address = load_address_32;
        info.mod_date = mod_date_32;
        pathname_address = pathname_address_32;
      } else {
        uint64_t load_address_64;
        uint64_t pathname_address_64;
        uint64_t mod_date_64;
        ::memcpy(&load_address_64, image_info_buf + (i * 3 * pointer_size), 8);
        ::memcpy(&pathname_address_64,
                 image_info_buf + (i * 3 * pointer_size) + pointer_size, 8);
        ::memcpy(&mod_date_64, image_info_buf + (i * 3 * pointer_size) +
                                   pointer_size + pointer_size,
                 8);
        info.load_address = load_address_64;
        info.mod_date = mod_date_64;
        pathname_address = pathname_address_64;
      }
      char strbuf[17];
      info.filename = "";
      uint64_t pathname_ptr = pathname_address;
      bool still_reading = true;
      while (still_reading &&
             ReadMemory(pathname_ptr, sizeof(strbuf) - 1, strbuf) ==
                 sizeof(strbuf) - 1) {
        strbuf[sizeof(strbuf) - 1] = '\0';
        info.filename += strbuf;
        pathname_ptr += sizeof(strbuf) - 1;
        // Stop if we found nul byte indicating the end of the string
        for (size_t i = 0; i < sizeof(strbuf) - 1; i++) {
          if (strbuf[i] == '\0') {
            still_reading = false;
            break;
          }
        }
      }
      uuid_clear(info.macho_info.uuid);
      image_infos.push_back(info);
    }
    if (image_infos.size() == 0) {
      return reply_sp;
    }

    free(image_info_buf);

    ////  Second, read the mach header / load commands for all the dylibs

    for (size_t i = 0; i < image_count; i++) {
      // The SPI to provide platform is not available on older systems.
      uint32_t platform = 0;
      if (!GetMachOInformationFromMemory(platform,
                                         image_infos[i].load_address,
                                         pointer_size,
                                         image_infos[i].macho_info)) {
        return reply_sp;
      }
    }

    ////  Third, format all of the above in the JSONGenerator object.

    return FormatDynamicLibrariesIntoJSON(image_infos);
  }

  return reply_sp;
}

// From dyld SPI header dyld_process_info.h
typedef void *dyld_process_info;
struct dyld_process_cache_info {
  uuid_t cacheUUID;          // UUID of cache used by process
  uint64_t cacheBaseAddress; // load address of dyld shared cache
  bool noCache;              // process is running without a dyld cache
  bool privateCache; // process is using a private copy of its dyld cache
};

// Use the dyld SPI present in macOS 10.12, iOS 10, tvOS 10, watchOS 3 and newer
// to get
// the load address, uuid, and filenames of all the libraries.
// This only fills in those three fields in the 'struct
// binary_image_information' - call
// GetMachOInformationFromMemory to fill in the mach-o header/load command
// details.
uint32_t MachProcess::GetAllLoadedBinariesViaDYLDSPI(
    std::vector<struct binary_image_information> &image_infos) {
  uint32_t platform = 0;
  kern_return_t kern_ret;
  if (m_dyld_process_info_create) {
    dyld_process_info info =
        m_dyld_process_info_create(m_task.TaskPort(), 0, &kern_ret);
    if (info) {
      m_dyld_process_info_for_each_image(
          info,
          ^(uint64_t mach_header_addr, const uuid_t uuid, const char *path) {
            struct binary_image_information image;
            image.filename = path;
            uuid_copy(image.macho_info.uuid, uuid);
            image.load_address = mach_header_addr;
            image_infos.push_back(image);
          });
      if (m_dyld_process_info_get_platform)
        platform = m_dyld_process_info_get_platform(info);
      m_dyld_process_info_release(info);
    }
  }
  return platform;
}

// Fetch information about all shared libraries using the dyld SPIs that exist
// in
// macOS 10.12, iOS 10, tvOS 10, watchOS 3 and newer.
JSONGenerator::ObjectSP
MachProcess::GetAllLoadedLibrariesInfos(nub_process_t pid) {
  JSONGenerator::DictionarySP reply_sp;

  int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PID, pid};
  struct kinfo_proc processInfo;
  size_t bufsize = sizeof(processInfo);
  if (sysctl(mib, (unsigned)(sizeof(mib) / sizeof(int)), &processInfo, &bufsize,
             NULL, 0) == 0 &&
      bufsize > 0) {
    uint32_t pointer_size = 4;
    if (processInfo.kp_proc.p_flag & P_LP64)
      pointer_size = 8;

    std::vector<struct binary_image_information> image_infos;
    uint32_t platform = GetAllLoadedBinariesViaDYLDSPI(image_infos);
    const size_t image_count = image_infos.size();
    for (size_t i = 0; i < image_count; i++) {
      GetMachOInformationFromMemory(platform,
                                    image_infos[i].load_address, pointer_size,
                                    image_infos[i].macho_info);
    }
    return FormatDynamicLibrariesIntoJSON(image_infos);
  }
  return reply_sp;
}

// Fetch information about the shared libraries at the given load addresses
// using the
// dyld SPIs that exist in macOS 10.12, iOS 10, tvOS 10, watchOS 3 and newer.
JSONGenerator::ObjectSP MachProcess::GetLibrariesInfoForAddresses(
    nub_process_t pid, std::vector<uint64_t> &macho_addresses) {
  JSONGenerator::DictionarySP reply_sp;

  int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PID, pid};
  struct kinfo_proc processInfo;
  size_t bufsize = sizeof(processInfo);
  if (sysctl(mib, (unsigned)(sizeof(mib) / sizeof(int)), &processInfo, &bufsize,
             NULL, 0) == 0 &&
      bufsize > 0) {
    uint32_t pointer_size = 4;
    if (processInfo.kp_proc.p_flag & P_LP64)
      pointer_size = 8;

    std::vector<struct binary_image_information> all_image_infos;
    uint32_t platform = GetAllLoadedBinariesViaDYLDSPI(all_image_infos);

    std::vector<struct binary_image_information> image_infos;
    const size_t macho_addresses_count = macho_addresses.size();
    const size_t all_image_infos_count = all_image_infos.size();
    for (size_t i = 0; i < macho_addresses_count; i++) {
      for (size_t j = 0; j < all_image_infos_count; j++) {
        if (all_image_infos[j].load_address == macho_addresses[i]) {
          image_infos.push_back(all_image_infos[j]);
        }
      }
    }

    const size_t image_infos_count = image_infos.size();
    for (size_t i = 0; i < image_infos_count; i++) {
      GetMachOInformationFromMemory(platform,
                                    image_infos[i].load_address, pointer_size,
                                    image_infos[i].macho_info);
    }
    return FormatDynamicLibrariesIntoJSON(image_infos);
  }
  return reply_sp;
}

// From dyld's internal podyld_process_info.h:

JSONGenerator::ObjectSP MachProcess::GetSharedCacheInfo(nub_process_t pid) {
  JSONGenerator::DictionarySP reply_sp(new JSONGenerator::Dictionary());
  ;
  kern_return_t kern_ret;
  if (m_dyld_process_info_create && m_dyld_process_info_get_cache) {
    dyld_process_info info =
        m_dyld_process_info_create(m_task.TaskPort(), 0, &kern_ret);
    if (info) {
      struct dyld_process_cache_info shared_cache_info;
      m_dyld_process_info_get_cache(info, &shared_cache_info);

      reply_sp->AddIntegerItem("shared_cache_base_address",
                               shared_cache_info.cacheBaseAddress);

      uuid_string_t uuidstr;
      uuid_unparse_upper(shared_cache_info.cacheUUID, uuidstr);
      reply_sp->AddStringItem("shared_cache_uuid", uuidstr);

      reply_sp->AddBooleanItem("no_shared_cache", shared_cache_info.noCache);
      reply_sp->AddBooleanItem("shared_cache_private_cache",
                               shared_cache_info.privateCache);

      m_dyld_process_info_release(info);
    }
  }
  return reply_sp;
}

nub_thread_t MachProcess::GetCurrentThread() {
  return m_thread_list.CurrentThreadID();
}

nub_thread_t MachProcess::GetCurrentThreadMachPort() {
  return m_thread_list.GetMachPortNumberByThreadID(
      m_thread_list.CurrentThreadID());
}

nub_thread_t MachProcess::SetCurrentThread(nub_thread_t tid) {
  return m_thread_list.SetCurrentThread(tid);
}

bool MachProcess::GetThreadStoppedReason(nub_thread_t tid,
                                         struct DNBThreadStopInfo *stop_info) {
  if (m_thread_list.GetThreadStoppedReason(tid, stop_info)) {
    if (m_did_exec)
      stop_info->reason = eStopTypeExec;
    return true;
  }
  return false;
}

void MachProcess::DumpThreadStoppedReason(nub_thread_t tid) const {
  return m_thread_list.DumpThreadStoppedReason(tid);
}

const char *MachProcess::GetThreadInfo(nub_thread_t tid) const {
  return m_thread_list.GetThreadInfo(tid);
}

uint32_t MachProcess::GetCPUType() {
  if (m_cpu_type == 0 && m_pid != 0)
    m_cpu_type = MachProcess::GetCPUTypeForLocalProcess(m_pid);
  return m_cpu_type;
}

const DNBRegisterSetInfo *
MachProcess::GetRegisterSetInfo(nub_thread_t tid,
                                nub_size_t *num_reg_sets) const {
  MachThreadSP thread_sp(m_thread_list.GetThreadByID(tid));
  if (thread_sp) {
    DNBArchProtocol *arch = thread_sp->GetArchProtocol();
    if (arch)
      return arch->GetRegisterSetInfo(num_reg_sets);
  }
  *num_reg_sets = 0;
  return NULL;
}

bool MachProcess::GetRegisterValue(nub_thread_t tid, uint32_t set, uint32_t reg,
                                   DNBRegisterValue *value) const {
  return m_thread_list.GetRegisterValue(tid, set, reg, value);
}

bool MachProcess::SetRegisterValue(nub_thread_t tid, uint32_t set, uint32_t reg,
                                   const DNBRegisterValue *value) const {
  return m_thread_list.SetRegisterValue(tid, set, reg, value);
}

void MachProcess::SetState(nub_state_t new_state) {
  // If any other threads access this we will need a mutex for it
  uint32_t event_mask = 0;

  // Scope for mutex locker
  {
    PTHREAD_MUTEX_LOCKER(locker, m_state_mutex);
    const nub_state_t old_state = m_state;

    if (old_state == eStateExited) {
      DNBLogThreadedIf(LOG_PROCESS, "MachProcess::SetState(%s) ignoring new "
                                    "state since current state is exited",
                       DNBStateAsString(new_state));
    } else if (old_state == new_state) {
      DNBLogThreadedIf(
          LOG_PROCESS,
          "MachProcess::SetState(%s) ignoring redundant state change...",
          DNBStateAsString(new_state));
    } else {
      if (NUB_STATE_IS_STOPPED(new_state))
        event_mask = eEventProcessStoppedStateChanged;
      else
        event_mask = eEventProcessRunningStateChanged;

      DNBLogThreadedIf(
          LOG_PROCESS, "MachProcess::SetState(%s) upating state (previous "
                       "state was %s), event_mask = 0x%8.8x",
          DNBStateAsString(new_state), DNBStateAsString(old_state), event_mask);

      m_state = new_state;
      if (new_state == eStateStopped)
        m_stop_count++;
    }
  }

  if (event_mask != 0) {
    m_events.SetEvents(event_mask);
    m_private_events.SetEvents(event_mask);
    if (event_mask == eEventProcessStoppedStateChanged)
      m_private_events.ResetEvents(eEventProcessRunningStateChanged);
    else
      m_private_events.ResetEvents(eEventProcessStoppedStateChanged);

    // Wait for the event bit to reset if a reset ACK is requested
    m_events.WaitForResetAck(event_mask);
  }
}

void MachProcess::Clear(bool detaching) {
  // Clear any cached thread list while the pid and task are still valid

  m_task.Clear();
  // Now clear out all member variables
  m_pid = INVALID_NUB_PROCESS;
  if (!detaching)
    CloseChildFileDescriptors();

  m_path.clear();
  m_args.clear();
  SetState(eStateUnloaded);
  m_flags = eMachProcessFlagsNone;
  m_stop_count = 0;
  m_thread_list.Clear();
  {
    PTHREAD_MUTEX_LOCKER(locker, m_exception_messages_mutex);
    m_exception_messages.clear();
  }
  m_activities.Clear();
  StopProfileThread();
}

bool MachProcess::StartSTDIOThread() {
  DNBLogThreadedIf(LOG_PROCESS, "MachProcess::%s ( )", __FUNCTION__);
  // Create the thread that watches for the child STDIO
  return ::pthread_create(&m_stdio_thread, NULL, MachProcess::STDIOThread,
                          this) == 0;
}

void MachProcess::SetEnableAsyncProfiling(bool enable, uint64_t interval_usec,
                                          DNBProfileDataScanType scan_type) {
  m_profile_enabled = enable;
  m_profile_interval_usec = static_cast<useconds_t>(interval_usec);
  m_profile_scan_type = scan_type;

  if (m_profile_enabled && (m_profile_thread == NULL)) {
    StartProfileThread();
  } else if (!m_profile_enabled && m_profile_thread) {
    StopProfileThread();
  }
}

void MachProcess::StopProfileThread() {
  if (m_profile_thread == NULL)
    return;
  m_profile_events.SetEvents(eMachProcessProfileCancel);
  pthread_join(m_profile_thread, NULL);
  m_profile_thread = NULL;
  m_profile_events.ResetEvents(eMachProcessProfileCancel);
}

bool MachProcess::StartProfileThread() {
  DNBLogThreadedIf(LOG_PROCESS, "MachProcess::%s ( )", __FUNCTION__);
  // Create the thread that profiles the inferior and reports back if enabled
  return ::pthread_create(&m_profile_thread, NULL, MachProcess::ProfileThread,
                          this) == 0;
}

nub_addr_t MachProcess::LookupSymbol(const char *name, const char *shlib) {
  if (m_name_to_addr_callback != NULL && name && name[0])
    return m_name_to_addr_callback(ProcessID(), name, shlib,
                                   m_name_to_addr_baton);
  return INVALID_NUB_ADDRESS;
}

bool MachProcess::Resume(const DNBThreadResumeActions &thread_actions) {
  DNBLogThreadedIf(LOG_PROCESS, "MachProcess::Resume ()");
  nub_state_t state = GetState();

  if (CanResume(state)) {
    m_thread_actions = thread_actions;
    PrivateResume();
    return true;
  } else if (state == eStateRunning) {
    DNBLog("Resume() - task 0x%x is already running, ignoring...",
           m_task.TaskPort());
    return true;
  }
  DNBLog("Resume() - task 0x%x has state %s, can't continue...",
         m_task.TaskPort(), DNBStateAsString(state));
  return false;
}

bool MachProcess::Kill(const struct timespec *timeout_abstime) {
  DNBLogThreadedIf(LOG_PROCESS, "MachProcess::Kill ()");
  nub_state_t state = DoSIGSTOP(true, false, NULL);
  DNBLogThreadedIf(LOG_PROCESS, "MachProcess::Kill() DoSIGSTOP() state = %s",
                   DNBStateAsString(state));
  errno = 0;
  DNBLog("Sending ptrace PT_KILL to terminate inferior process.");
  ::ptrace(PT_KILL, m_pid, 0, 0);
  DNBError err;
  err.SetErrorToErrno();
  if (DNBLogCheckLogBit(LOG_PROCESS) || err.Fail()) {
    err.LogThreaded("MachProcess::Kill() DoSIGSTOP() ::ptrace "
            "(PT_KILL, pid=%u, 0, 0) => 0x%8.8x (%s)",
            m_pid, err.Status(), err.AsString());
  }
  m_thread_actions = DNBThreadResumeActions(eStateRunning, 0);
  PrivateResume();

  // Try and reap the process without touching our m_events since
  // we want the code above this to still get the eStateExited event
  const uint32_t reap_timeout_usec =
      1000000; // Wait 1 second and try to reap the process
  const uint32_t reap_interval_usec = 10000; //
  uint32_t reap_time_elapsed;
  for (reap_time_elapsed = 0; reap_time_elapsed < reap_timeout_usec;
       reap_time_elapsed += reap_interval_usec) {
    if (GetState() == eStateExited)
      break;
    usleep(reap_interval_usec);
  }
  DNBLog("Waited %u ms for process to be reaped (state = %s)",
         reap_time_elapsed / 1000, DNBStateAsString(GetState()));
  return true;
}

bool MachProcess::Interrupt() {
  nub_state_t state = GetState();
  if (IsRunning(state)) {
    if (m_sent_interrupt_signo == 0) {
      m_sent_interrupt_signo = SIGSTOP;
      if (Signal(m_sent_interrupt_signo)) {
        DNBLogThreadedIf(
            LOG_PROCESS,
            "MachProcess::Interrupt() - sent %i signal to interrupt process",
            m_sent_interrupt_signo);
        return true;
      } else {
        m_sent_interrupt_signo = 0;
        DNBLogThreadedIf(LOG_PROCESS, "MachProcess::Interrupt() - failed to "
                                      "send %i signal to interrupt process",
                         m_sent_interrupt_signo);
      }
    } else {
      DNBLogThreadedIf(LOG_PROCESS, "MachProcess::Interrupt() - previously "
                                    "sent an interrupt signal %i that hasn't "
                                    "been received yet, interrupt aborted",
                       m_sent_interrupt_signo);
    }
  } else {
    DNBLogThreadedIf(LOG_PROCESS, "MachProcess::Interrupt() - process already "
                                  "stopped, no interrupt sent");
  }
  return false;
}

bool MachProcess::Signal(int signal, const struct timespec *timeout_abstime) {
  DNBLogThreadedIf(LOG_PROCESS,
                   "MachProcess::Signal (signal = %d, timeout = %p)", signal,
                   static_cast<const void *>(timeout_abstime));
  nub_state_t state = GetState();
  if (::kill(ProcessID(), signal) == 0) {
    // If we were running and we have a timeout, wait for the signal to stop
    if (IsRunning(state) && timeout_abstime) {
      DNBLogThreadedIf(LOG_PROCESS,
                       "MachProcess::Signal (signal = %d, timeout "
                       "= %p) waiting for signal to stop "
                       "process...",
                       signal, static_cast<const void *>(timeout_abstime));
      m_private_events.WaitForSetEvents(eEventProcessStoppedStateChanged,
                                        timeout_abstime);
      state = GetState();
      DNBLogThreadedIf(
          LOG_PROCESS,
          "MachProcess::Signal (signal = %d, timeout = %p) state = %s", signal,
          static_cast<const void *>(timeout_abstime), DNBStateAsString(state));
      return !IsRunning(state);
    }
    DNBLogThreadedIf(
        LOG_PROCESS,
        "MachProcess::Signal (signal = %d, timeout = %p) not waiting...",
        signal, static_cast<const void *>(timeout_abstime));
    return true;
  }
  DNBError err(errno, DNBError::POSIX);
  err.LogThreadedIfError("kill (pid = %d, signo = %i)", ProcessID(), signal);
  return false;
}

bool MachProcess::SendEvent(const char *event, DNBError &send_err) {
  DNBLogThreadedIf(LOG_PROCESS,
                   "MachProcess::SendEvent (event = %s) to pid: %d", event,
                   m_pid);
  if (m_pid == INVALID_NUB_PROCESS)
    return false;
// FIXME: Shouldn't we use the launch flavor we were started with?
#if defined(WITH_FBS) || defined(WITH_BKS)
  return BoardServiceSendEvent(event, send_err);
#endif
  return true;
}

nub_state_t MachProcess::DoSIGSTOP(bool clear_bps_and_wps, bool allow_running,
                                   uint32_t *thread_idx_ptr) {
  nub_state_t state = GetState();
  DNBLogThreadedIf(LOG_PROCESS, "MachProcess::DoSIGSTOP() state = %s",
                   DNBStateAsString(state));

  if (!IsRunning(state)) {
    if (clear_bps_and_wps) {
      DisableAllBreakpoints(true);
      DisableAllWatchpoints(true);
      clear_bps_and_wps = false;
    }

    // If we already have a thread stopped due to a SIGSTOP, we don't have
    // to do anything...
    uint32_t thread_idx =
        m_thread_list.GetThreadIndexForThreadStoppedWithSignal(SIGSTOP);
    if (thread_idx_ptr)
      *thread_idx_ptr = thread_idx;
    if (thread_idx != UINT32_MAX)
      return GetState();

    // No threads were stopped with a SIGSTOP, we need to run and halt the
    // process with a signal
    DNBLogThreadedIf(LOG_PROCESS,
                     "MachProcess::DoSIGSTOP() state = %s -- resuming process",
                     DNBStateAsString(state));
    if (allow_running)
      m_thread_actions = DNBThreadResumeActions(eStateRunning, 0);
    else
      m_thread_actions = DNBThreadResumeActions(eStateSuspended, 0);

    PrivateResume();

    // Reset the event that says we were indeed running
    m_events.ResetEvents(eEventProcessRunningStateChanged);
    state = GetState();
  }

  // We need to be stopped in order to be able to detach, so we need
  // to send ourselves a SIGSTOP

  DNBLogThreadedIf(LOG_PROCESS,
                   "MachProcess::DoSIGSTOP() state = %s -- sending SIGSTOP",
                   DNBStateAsString(state));
  struct timespec sigstop_timeout;
  DNBTimer::OffsetTimeOfDay(&sigstop_timeout, 2, 0);
  Signal(SIGSTOP, &sigstop_timeout);
  if (clear_bps_and_wps) {
    DisableAllBreakpoints(true);
    DisableAllWatchpoints(true);
    // clear_bps_and_wps = false;
  }
  uint32_t thread_idx =
      m_thread_list.GetThreadIndexForThreadStoppedWithSignal(SIGSTOP);
  if (thread_idx_ptr)
    *thread_idx_ptr = thread_idx;
  return GetState();
}

bool MachProcess::Detach() {
  DNBLogThreadedIf(LOG_PROCESS, "MachProcess::Detach()");

  uint32_t thread_idx = UINT32_MAX;
  nub_state_t state = DoSIGSTOP(true, true, &thread_idx);
  DNBLogThreadedIf(LOG_PROCESS, "MachProcess::Detach() DoSIGSTOP() returned %s",
                   DNBStateAsString(state));

  {
    m_thread_actions.Clear();
    m_activities.Clear();
    DNBThreadResumeAction thread_action;
    thread_action.tid = m_thread_list.ThreadIDAtIndex(thread_idx);
    thread_action.state = eStateRunning;
    thread_action.signal = -1;
    thread_action.addr = INVALID_NUB_ADDRESS;

    m_thread_actions.Append(thread_action);
    m_thread_actions.SetDefaultThreadActionIfNeeded(eStateRunning, 0);

    PTHREAD_MUTEX_LOCKER(locker, m_exception_messages_mutex);

    ReplyToAllExceptions();
  }

  m_task.ShutDownExcecptionThread();

  // Detach from our process
  errno = 0;
  nub_process_t pid = m_pid;
  int ret = ::ptrace(PT_DETACH, pid, (caddr_t)1, 0);
  DNBError err(errno, DNBError::POSIX);
  if (DNBLogCheckLogBit(LOG_PROCESS) || err.Fail() || (ret != 0))
    err.LogThreaded("::ptrace (PT_DETACH, %u, (caddr_t)1, 0)", pid);

  // Resume our task
  m_task.Resume();

  // NULL our task out as we have already restored all exception ports
  m_task.Clear();

  // Clear out any notion of the process we once were
  const bool detaching = true;
  Clear(detaching);

  SetState(eStateDetached);

  return true;
}

//----------------------------------------------------------------------
// ReadMemory from the MachProcess level will always remove any software
// breakpoints from the memory buffer before returning. If you wish to
// read memory and see those traps, read from the MachTask
// (m_task.ReadMemory()) as that version will give you what is actually
// in inferior memory.
//----------------------------------------------------------------------
nub_size_t MachProcess::ReadMemory(nub_addr_t addr, nub_size_t size,
                                   void *buf) {
  // We need to remove any current software traps (enabled software
  // breakpoints) that we may have placed in our tasks memory.

  // First just read the memory as is
  nub_size_t bytes_read = m_task.ReadMemory(addr, size, buf);

  // Then place any opcodes that fall into this range back into the buffer
  // before we return this to callers.
  if (bytes_read > 0)
    m_breakpoints.RemoveTrapsFromBuffer(addr, bytes_read, buf);
  return bytes_read;
}

//----------------------------------------------------------------------
// WriteMemory from the MachProcess level will always write memory around
// any software breakpoints. Any software breakpoints will have their
// opcodes modified if they are enabled. Any memory that doesn't overlap
// with software breakpoints will be written to. If you wish to write to
// inferior memory without this interference, then write to the MachTask
// (m_task.WriteMemory()) as that version will always modify inferior
// memory.
//----------------------------------------------------------------------
nub_size_t MachProcess::WriteMemory(nub_addr_t addr, nub_size_t size,
                                    const void *buf) {
  // We need to write any data that would go where any current software traps
  // (enabled software breakpoints) any software traps (breakpoints) that we
  // may have placed in our tasks memory.

  std::vector<DNBBreakpoint *> bps;

  const size_t num_bps =
      m_breakpoints.FindBreakpointsThatOverlapRange(addr, size, bps);
  if (num_bps == 0)
    return m_task.WriteMemory(addr, size, buf);

  nub_size_t bytes_written = 0;
  nub_addr_t intersect_addr;
  nub_size_t intersect_size;
  nub_size_t opcode_offset;
  const uint8_t *ubuf = (const uint8_t *)buf;

  for (size_t i = 0; i < num_bps; ++i) {
    DNBBreakpoint *bp = bps[i];

    const bool intersects = bp->IntersectsRange(
        addr, size, &intersect_addr, &intersect_size, &opcode_offset);
    UNUSED_IF_ASSERT_DISABLED(intersects);
    assert(intersects);
    assert(addr <= intersect_addr && intersect_addr < addr + size);
    assert(addr < intersect_addr + intersect_size &&
           intersect_addr + intersect_size <= addr + size);
    assert(opcode_offset + intersect_size <= bp->ByteSize());

    // Check for bytes before this breakpoint
    const nub_addr_t curr_addr = addr + bytes_written;
    if (intersect_addr > curr_addr) {
      // There are some bytes before this breakpoint that we need to
      // just write to memory
      nub_size_t curr_size = intersect_addr - curr_addr;
      nub_size_t curr_bytes_written =
          m_task.WriteMemory(curr_addr, curr_size, ubuf + bytes_written);
      bytes_written += curr_bytes_written;
      if (curr_bytes_written != curr_size) {
        // We weren't able to write all of the requested bytes, we
        // are done looping and will return the number of bytes that
        // we have written so far.
        break;
      }
    }

    // Now write any bytes that would cover up any software breakpoints
    // directly into the breakpoint opcode buffer
    ::memcpy(bp->SavedOpcodeBytes() + opcode_offset, ubuf + bytes_written,
             intersect_size);
    bytes_written += intersect_size;
  }

  // Write any remaining bytes after the last breakpoint if we have any left
  if (bytes_written < size)
    bytes_written += m_task.WriteMemory(
        addr + bytes_written, size - bytes_written, ubuf + bytes_written);

  return bytes_written;
}

void MachProcess::ReplyToAllExceptions() {
  PTHREAD_MUTEX_LOCKER(locker, m_exception_messages_mutex);
  if (!m_exception_messages.empty()) {
    MachException::Message::iterator pos;
    MachException::Message::iterator begin = m_exception_messages.begin();
    MachException::Message::iterator end = m_exception_messages.end();
    for (pos = begin; pos != end; ++pos) {
      DNBLogThreadedIf(LOG_EXCEPTIONS, "Replying to exception %u...",
                       (uint32_t)std::distance(begin, pos));
      int thread_reply_signal = 0;

      nub_thread_t tid =
          m_thread_list.GetThreadIDByMachPortNumber(pos->state.thread_port);
      const DNBThreadResumeAction *action = NULL;
      if (tid != INVALID_NUB_THREAD) {
        action = m_thread_actions.GetActionForThread(tid, false);
      }

      if (action) {
        thread_reply_signal = action->signal;
        if (thread_reply_signal)
          m_thread_actions.SetSignalHandledForThread(tid);
      }

      DNBError err(pos->Reply(this, thread_reply_signal));
      if (DNBLogCheckLogBit(LOG_EXCEPTIONS))
        err.LogThreadedIfError("Error replying to exception");
    }

    // Erase all exception message as we should have used and replied
    // to them all already.
    m_exception_messages.clear();
  }
}
void MachProcess::PrivateResume() {
  PTHREAD_MUTEX_LOCKER(locker, m_exception_messages_mutex);

  m_auto_resume_signo = m_sent_interrupt_signo;
  if (m_auto_resume_signo)
    DNBLogThreadedIf(LOG_PROCESS, "MachProcess::PrivateResume() - task 0x%x "
                                  "resuming (with unhandled interrupt signal "
                                  "%i)...",
                     m_task.TaskPort(), m_auto_resume_signo);
  else
    DNBLogThreadedIf(LOG_PROCESS,
                     "MachProcess::PrivateResume() - task 0x%x resuming...",
                     m_task.TaskPort());

  ReplyToAllExceptions();
  //    bool stepOverBreakInstruction = step;

  // Let the thread prepare to resume and see if any threads want us to
  // step over a breakpoint instruction (ProcessWillResume will modify
  // the value of stepOverBreakInstruction).
  m_thread_list.ProcessWillResume(this, m_thread_actions);

  // Set our state accordingly
  if (m_thread_actions.NumActionsWithState(eStateStepping))
    SetState(eStateStepping);
  else
    SetState(eStateRunning);

  // Now resume our task.
  m_task.Resume();
}

DNBBreakpoint *MachProcess::CreateBreakpoint(nub_addr_t addr, nub_size_t length,
                                             bool hardware) {
  DNBLogThreadedIf(LOG_BREAKPOINTS, "MachProcess::CreateBreakpoint ( addr = "
                                    "0x%8.8llx, length = %llu, hardware = %i)",
                   (uint64_t)addr, (uint64_t)length, hardware);

  DNBBreakpoint *bp = m_breakpoints.FindByAddress(addr);
  if (bp)
    bp->Retain();
  else
    bp = m_breakpoints.Add(addr, length, hardware);

  if (EnableBreakpoint(addr)) {
    DNBLogThreadedIf(LOG_BREAKPOINTS,
                     "MachProcess::CreateBreakpoint ( addr = "
                     "0x%8.8llx, length = %llu) => %p",
                     (uint64_t)addr, (uint64_t)length, static_cast<void *>(bp));
    return bp;
  } else if (bp->Release() == 0) {
    m_breakpoints.Remove(addr);
  }
  // We failed to enable the breakpoint
  return NULL;
}

DNBBreakpoint *MachProcess::CreateWatchpoint(nub_addr_t addr, nub_size_t length,
                                             uint32_t watch_flags,
                                             bool hardware) {
  DNBLogThreadedIf(LOG_WATCHPOINTS, "MachProcess::CreateWatchpoint ( addr = "
                                    "0x%8.8llx, length = %llu, flags = "
                                    "0x%8.8x, hardware = %i)",
                   (uint64_t)addr, (uint64_t)length, watch_flags, hardware);

  DNBBreakpoint *wp = m_watchpoints.FindByAddress(addr);
  // since the Z packets only send an address, we can only have one watchpoint
  // at
  // an address. If there is already one, we must refuse to create another
  // watchpoint
  if (wp)
    return NULL;

  wp = m_watchpoints.Add(addr, length, hardware);
  wp->SetIsWatchpoint(watch_flags);

  if (EnableWatchpoint(addr)) {
    DNBLogThreadedIf(LOG_WATCHPOINTS,
                     "MachProcess::CreateWatchpoint ( addr = "
                     "0x%8.8llx, length = %llu) => %p",
                     (uint64_t)addr, (uint64_t)length, static_cast<void *>(wp));
    return wp;
  } else {
    DNBLogThreadedIf(LOG_WATCHPOINTS, "MachProcess::CreateWatchpoint ( addr = "
                                      "0x%8.8llx, length = %llu) => FAILED",
                     (uint64_t)addr, (uint64_t)length);
    m_watchpoints.Remove(addr);
  }
  // We failed to enable the watchpoint
  return NULL;
}

void MachProcess::DisableAllBreakpoints(bool remove) {
  DNBLogThreadedIf(LOG_BREAKPOINTS, "MachProcess::%s (remove = %d )",
                   __FUNCTION__, remove);

  m_breakpoints.DisableAllBreakpoints(this);

  if (remove)
    m_breakpoints.RemoveDisabled();
}

void MachProcess::DisableAllWatchpoints(bool remove) {
  DNBLogThreadedIf(LOG_WATCHPOINTS, "MachProcess::%s (remove = %d )",
                   __FUNCTION__, remove);

  m_watchpoints.DisableAllWatchpoints(this);

  if (remove)
    m_watchpoints.RemoveDisabled();
}

bool MachProcess::DisableBreakpoint(nub_addr_t addr, bool remove) {
  DNBBreakpoint *bp = m_breakpoints.FindByAddress(addr);
  if (bp) {
    // After "exec" we might end up with a bunch of breakpoints that were
    // disabled
    // manually, just ignore them
    if (!bp->IsEnabled()) {
      // Breakpoint might have been disabled by an exec
      if (remove && bp->Release() == 0) {
        m_thread_list.NotifyBreakpointChanged(bp);
        m_breakpoints.Remove(addr);
      }
      return true;
    }

    // We have multiple references to this breakpoint, decrement the ref count
    // and if it isn't zero, then return true;
    if (remove && bp->Release() > 0)
      return true;

    DNBLogThreadedIf(
        LOG_BREAKPOINTS | LOG_VERBOSE,
        "MachProcess::DisableBreakpoint ( addr = 0x%8.8llx, remove = %d )",
        (uint64_t)addr, remove);

    if (bp->IsHardware()) {
      bool hw_disable_result = m_thread_list.DisableHardwareBreakpoint(bp);

      if (hw_disable_result) {
        bp->SetEnabled(false);
        // Let the thread list know that a breakpoint has been modified
        if (remove) {
          m_thread_list.NotifyBreakpointChanged(bp);
          m_breakpoints.Remove(addr);
        }
        DNBLogThreadedIf(LOG_BREAKPOINTS, "MachProcess::DisableBreakpoint ( "
                                          "addr = 0x%8.8llx, remove = %d ) "
                                          "(hardware) => success",
                         (uint64_t)addr, remove);
        return true;
      }

      return false;
    }

    const nub_size_t break_op_size = bp->ByteSize();
    assert(break_op_size > 0);
    const uint8_t *const break_op =
        DNBArchProtocol::GetBreakpointOpcode(bp->ByteSize());
    if (break_op_size > 0) {
      // Clear a software breakpoint instruction
      uint8_t curr_break_op[break_op_size];
      bool break_op_found = false;

      // Read the breakpoint opcode
      if (m_task.ReadMemory(addr, break_op_size, curr_break_op) ==
          break_op_size) {
        bool verify = false;
        if (bp->IsEnabled()) {
          // Make sure a breakpoint opcode exists at this address
          if (memcmp(curr_break_op, break_op, break_op_size) == 0) {
            break_op_found = true;
            // We found a valid breakpoint opcode at this address, now restore
            // the saved opcode.
            if (m_task.WriteMemory(addr, break_op_size,
                                   bp->SavedOpcodeBytes()) == break_op_size) {
              verify = true;
            } else {
              DNBLogError("MachProcess::DisableBreakpoint ( addr = 0x%8.8llx, "
                          "remove = %d ) memory write failed when restoring "
                          "original opcode",
                          (uint64_t)addr, remove);
            }
          } else {
            DNBLogWarning("MachProcess::DisableBreakpoint ( addr = 0x%8.8llx, "
                          "remove = %d ) expected a breakpoint opcode but "
                          "didn't find one.",
                          (uint64_t)addr, remove);
            // Set verify to true and so we can check if the original opcode has
            // already been restored
            verify = true;
          }
        } else {
          DNBLogThreadedIf(LOG_BREAKPOINTS | LOG_VERBOSE,
                           "MachProcess::DisableBreakpoint ( addr = 0x%8.8llx, "
                           "remove = %d ) is not enabled",
                           (uint64_t)addr, remove);
          // Set verify to true and so we can check if the original opcode is
          // there
          verify = true;
        }

        if (verify) {
          uint8_t verify_opcode[break_op_size];
          // Verify that our original opcode made it back to the inferior
          if (m_task.ReadMemory(addr, break_op_size, verify_opcode) ==
              break_op_size) {
            // compare the memory we just read with the original opcode
            if (memcmp(bp->SavedOpcodeBytes(), verify_opcode, break_op_size) ==
                0) {
              // SUCCESS
              bp->SetEnabled(false);
              // Let the thread list know that a breakpoint has been modified
              if (remove && bp->Release() == 0) {
                m_thread_list.NotifyBreakpointChanged(bp);
                m_breakpoints.Remove(addr);
              }
              DNBLogThreadedIf(LOG_BREAKPOINTS,
                               "MachProcess::DisableBreakpoint ( addr = "
                               "0x%8.8llx, remove = %d ) => success",
                               (uint64_t)addr, remove);
              return true;
            } else {
              if (break_op_found)
                DNBLogError("MachProcess::DisableBreakpoint ( addr = "
                            "0x%8.8llx, remove = %d ) : failed to restore "
                            "original opcode",
                            (uint64_t)addr, remove);
              else
                DNBLogError("MachProcess::DisableBreakpoint ( addr = "
                            "0x%8.8llx, remove = %d ) : opcode changed",
                            (uint64_t)addr, remove);
            }
          } else {
            DNBLogWarning("MachProcess::DisableBreakpoint: unable to disable "
                          "breakpoint 0x%8.8llx",
                          (uint64_t)addr);
          }
        }
      } else {
        DNBLogWarning("MachProcess::DisableBreakpoint: unable to read memory "
                      "at 0x%8.8llx",
                      (uint64_t)addr);
      }
    }
  } else {
    DNBLogError("MachProcess::DisableBreakpoint ( addr = 0x%8.8llx, remove = "
                "%d ) invalid breakpoint address",
                (uint64_t)addr, remove);
  }
  return false;
}

bool MachProcess::DisableWatchpoint(nub_addr_t addr, bool remove) {
  DNBLogThreadedIf(LOG_WATCHPOINTS,
                   "MachProcess::%s(addr = 0x%8.8llx, remove = %d)",
                   __FUNCTION__, (uint64_t)addr, remove);
  DNBBreakpoint *wp = m_watchpoints.FindByAddress(addr);
  if (wp) {
    // If we have multiple references to a watchpoint, removing the watchpoint
    // shouldn't clear it
    if (remove && wp->Release() > 0)
      return true;

    nub_addr_t addr = wp->Address();
    DNBLogThreadedIf(
        LOG_WATCHPOINTS,
        "MachProcess::DisableWatchpoint ( addr = 0x%8.8llx, remove = %d )",
        (uint64_t)addr, remove);

    if (wp->IsHardware()) {
      bool hw_disable_result = m_thread_list.DisableHardwareWatchpoint(wp);

      if (hw_disable_result) {
        wp->SetEnabled(false);
        if (remove)
          m_watchpoints.Remove(addr);
        DNBLogThreadedIf(LOG_WATCHPOINTS, "MachProcess::Disablewatchpoint ( "
                                          "addr = 0x%8.8llx, remove = %d ) "
                                          "(hardware) => success",
                         (uint64_t)addr, remove);
        return true;
      }
    }

    // TODO: clear software watchpoints if we implement them
  } else {
    DNBLogError("MachProcess::DisableWatchpoint ( addr = 0x%8.8llx, remove = "
                "%d ) invalid watchpoint ID",
                (uint64_t)addr, remove);
  }
  return false;
}

uint32_t MachProcess::GetNumSupportedHardwareWatchpoints() const {
  return m_thread_list.NumSupportedHardwareWatchpoints();
}

bool MachProcess::EnableBreakpoint(nub_addr_t addr) {
  DNBLogThreadedIf(LOG_BREAKPOINTS,
                   "MachProcess::EnableBreakpoint ( addr = 0x%8.8llx )",
                   (uint64_t)addr);
  DNBBreakpoint *bp = m_breakpoints.FindByAddress(addr);
  if (bp) {
    if (bp->IsEnabled()) {
      DNBLogWarning("MachProcess::EnableBreakpoint ( addr = 0x%8.8llx ): "
                    "breakpoint already enabled.",
                    (uint64_t)addr);
      return true;
    } else {
      if (bp->HardwarePreferred()) {
        bp->SetHardwareIndex(m_thread_list.EnableHardwareBreakpoint(bp));
        if (bp->IsHardware()) {
          bp->SetEnabled(true);
          return true;
        }
      }

      const nub_size_t break_op_size = bp->ByteSize();
      assert(break_op_size != 0);
      const uint8_t *const break_op =
          DNBArchProtocol::GetBreakpointOpcode(break_op_size);
      if (break_op_size > 0) {
        // Save the original opcode by reading it
        if (m_task.ReadMemory(addr, break_op_size, bp->SavedOpcodeBytes()) ==
            break_op_size) {
          // Write a software breakpoint in place of the original opcode
          if (m_task.WriteMemory(addr, break_op_size, break_op) ==
              break_op_size) {
            uint8_t verify_break_op[4];
            if (m_task.ReadMemory(addr, break_op_size, verify_break_op) ==
                break_op_size) {
              if (memcmp(break_op, verify_break_op, break_op_size) == 0) {
                bp->SetEnabled(true);
                // Let the thread list know that a breakpoint has been modified
                m_thread_list.NotifyBreakpointChanged(bp);
                DNBLogThreadedIf(LOG_BREAKPOINTS, "MachProcess::"
                                                  "EnableBreakpoint ( addr = "
                                                  "0x%8.8llx ) : SUCCESS.",
                                 (uint64_t)addr);
                return true;
              } else {
                DNBLogError("MachProcess::EnableBreakpoint ( addr = 0x%8.8llx "
                            "): breakpoint opcode verification failed.",
                            (uint64_t)addr);
              }
            } else {
              DNBLogError("MachProcess::EnableBreakpoint ( addr = 0x%8.8llx ): "
                          "unable to read memory to verify breakpoint opcode.",
                          (uint64_t)addr);
            }
          } else {
            DNBLogError("MachProcess::EnableBreakpoint ( addr = 0x%8.8llx ): "
                        "unable to write breakpoint opcode to memory.",
                        (uint64_t)addr);
          }
        } else {
          DNBLogError("MachProcess::EnableBreakpoint ( addr = 0x%8.8llx ): "
                      "unable to read memory at breakpoint address.",
                      (uint64_t)addr);
        }
      } else {
        DNBLogError("MachProcess::EnableBreakpoint ( addr = 0x%8.8llx ) no "
                    "software breakpoint opcode for current architecture.",
                    (uint64_t)addr);
      }
    }
  }
  return false;
}

bool MachProcess::EnableWatchpoint(nub_addr_t addr) {
  DNBLogThreadedIf(LOG_WATCHPOINTS,
                   "MachProcess::EnableWatchpoint(addr = 0x%8.8llx)",
                   (uint64_t)addr);
  DNBBreakpoint *wp = m_watchpoints.FindByAddress(addr);
  if (wp) {
    nub_addr_t addr = wp->Address();
    if (wp->IsEnabled()) {
      DNBLogWarning("MachProcess::EnableWatchpoint(addr = 0x%8.8llx): "
                    "watchpoint already enabled.",
                    (uint64_t)addr);
      return true;
    } else {
      // Currently only try and set hardware watchpoints.
      wp->SetHardwareIndex(m_thread_list.EnableHardwareWatchpoint(wp));
      if (wp->IsHardware()) {
        wp->SetEnabled(true);
        return true;
      }
      // TODO: Add software watchpoints by doing page protection tricks.
    }
  }
  return false;
}

// Called by the exception thread when an exception has been received from
// our process. The exception message is completely filled and the exception
// data has already been copied.
void MachProcess::ExceptionMessageReceived(
    const MachException::Message &exceptionMessage) {
  PTHREAD_MUTEX_LOCKER(locker, m_exception_messages_mutex);

  if (m_exception_messages.empty())
    m_task.Suspend();

  DNBLogThreadedIf(LOG_EXCEPTIONS, "MachProcess::ExceptionMessageReceived ( )");

  // Use a locker to automatically unlock our mutex in case of exceptions
  // Add the exception to our internal exception stack
  m_exception_messages.push_back(exceptionMessage);
}

task_t MachProcess::ExceptionMessageBundleComplete() {
  // We have a complete bundle of exceptions for our child process.
  PTHREAD_MUTEX_LOCKER(locker, m_exception_messages_mutex);
  DNBLogThreadedIf(LOG_EXCEPTIONS, "%s: %llu exception messages.",
                   __PRETTY_FUNCTION__, (uint64_t)m_exception_messages.size());
  bool auto_resume = false;
  if (!m_exception_messages.empty()) {
    m_did_exec = false;
    // First check for any SIGTRAP and make sure we didn't exec
    const task_t task = m_task.TaskPort();
    size_t i;
    if (m_pid != 0) {
      bool received_interrupt = false;
      uint32_t num_task_exceptions = 0;
      for (i = 0; i < m_exception_messages.size(); ++i) {
        if (m_exception_messages[i].state.task_port == task) {
          ++num_task_exceptions;
          const int signo = m_exception_messages[i].state.SoftSignal();
          if (signo == SIGTRAP) {
            // SIGTRAP could mean that we exec'ed. We need to check the
            // dyld all_image_infos.infoArray to see if it is NULL and if
            // so, say that we exec'ed.
            const nub_addr_t aii_addr = GetDYLDAllImageInfosAddress();
            if (aii_addr != INVALID_NUB_ADDRESS) {
              const nub_addr_t info_array_count_addr = aii_addr + 4;
              uint32_t info_array_count = 0;
              if (m_task.ReadMemory(info_array_count_addr, 4,
                                    &info_array_count) == 4) {
                if (info_array_count == 0) {
                  m_did_exec = true;
                  // Force the task port to update itself in case the task port
                  // changed after exec
                  DNBError err;
                  const task_t old_task = m_task.TaskPort();
                  const task_t new_task =
                      m_task.TaskPortForProcessID(err, true);
                  if (old_task != new_task)
                    DNBLogThreadedIf(
                        LOG_PROCESS,
                        "exec: task changed from 0x%4.4x to 0x%4.4x", old_task,
                        new_task);
                }
              } else {
                DNBLog("error: failed to read all_image_infos.infoArrayCount "
                       "from 0x%8.8llx",
                       (uint64_t)info_array_count_addr);
              }
            }
            break;
          } else if (m_sent_interrupt_signo != 0 &&
                     signo == m_sent_interrupt_signo) {
            received_interrupt = true;
          }
        }
      }

      if (m_did_exec) {
        cpu_type_t process_cpu_type =
            MachProcess::GetCPUTypeForLocalProcess(m_pid);
        if (m_cpu_type != process_cpu_type) {
          DNBLog("arch changed from 0x%8.8x to 0x%8.8x", m_cpu_type,
                 process_cpu_type);
          m_cpu_type = process_cpu_type;
          DNBArchProtocol::SetArchitecture(process_cpu_type);
        }
        m_thread_list.Clear();
        m_activities.Clear();
        m_breakpoints.DisableAll();
      }

      if (m_sent_interrupt_signo != 0) {
        if (received_interrupt) {
          DNBLogThreadedIf(LOG_PROCESS,
                           "MachProcess::ExceptionMessageBundleComplete(): "
                           "process successfully interrupted with signal %i",
                           m_sent_interrupt_signo);

          // Mark that we received the interrupt signal
          m_sent_interrupt_signo = 0;
          // Not check if we had a case where:
          // 1 - We called MachProcess::Interrupt() but we stopped for another
          // reason
          // 2 - We called MachProcess::Resume() (but still haven't gotten the
          // interrupt signal)
          // 3 - We are now incorrectly stopped because we are handling the
          // interrupt signal we missed
          // 4 - We might need to resume if we stopped only with the interrupt
          // signal that we never handled
          if (m_auto_resume_signo != 0) {
            // Only auto_resume if we stopped with _only_ the interrupt signal
            if (num_task_exceptions == 1) {
              auto_resume = true;
              DNBLogThreadedIf(LOG_PROCESS, "MachProcess::"
                                            "ExceptionMessageBundleComplete(): "
                                            "auto resuming due to unhandled "
                                            "interrupt signal %i",
                               m_auto_resume_signo);
            }
            m_auto_resume_signo = 0;
          }
        } else {
          DNBLogThreadedIf(LOG_PROCESS, "MachProcess::"
                                        "ExceptionMessageBundleComplete(): "
                                        "didn't get signal %i after "
                                        "MachProcess::Interrupt()",
                           m_sent_interrupt_signo);
        }
      }
    }

    // Let all threads recover from stopping and do any clean up based
    // on the previous thread state (if any).
    m_thread_list.ProcessDidStop(this);
    m_activities.Clear();

    // Let each thread know of any exceptions
    for (i = 0; i < m_exception_messages.size(); ++i) {
      // Let the thread list figure use the MachProcess to forward all
      // exceptions
      // on down to each thread.
      if (m_exception_messages[i].state.task_port == task)
        m_thread_list.NotifyException(m_exception_messages[i].state);
      if (DNBLogCheckLogBit(LOG_EXCEPTIONS))
        m_exception_messages[i].Dump();
    }

    if (DNBLogCheckLogBit(LOG_THREAD))
      m_thread_list.Dump();

    bool step_more = false;
    if (m_thread_list.ShouldStop(step_more) && !auto_resume) {
      // Wait for the eEventProcessRunningStateChanged event to be reset
      // before changing state to stopped to avoid race condition with
      // very fast start/stops
      struct timespec timeout;
      // DNBTimer::OffsetTimeOfDay(&timeout, 0, 250 * 1000);   // Wait for 250
      // ms
      DNBTimer::OffsetTimeOfDay(&timeout, 1, 0); // Wait for 250 ms
      m_events.WaitForEventsToReset(eEventProcessRunningStateChanged, &timeout);
      SetState(eStateStopped);
    } else {
      // Resume without checking our current state.
      PrivateResume();
    }
  } else {
    DNBLogThreadedIf(
        LOG_EXCEPTIONS, "%s empty exception messages bundle (%llu exceptions).",
        __PRETTY_FUNCTION__, (uint64_t)m_exception_messages.size());
  }
  return m_task.TaskPort();
}

nub_size_t
MachProcess::CopyImageInfos(struct DNBExecutableImageInfo **image_infos,
                            bool only_changed) {
  if (m_image_infos_callback != NULL)
    return m_image_infos_callback(ProcessID(), image_infos, only_changed,
                                  m_image_infos_baton);
  return 0;
}

void MachProcess::SharedLibrariesUpdated() {
  uint32_t event_bits = eEventSharedLibsStateChange;
  // Set the shared library event bit to let clients know of shared library
  // changes
  m_events.SetEvents(event_bits);
  // Wait for the event bit to reset if a reset ACK is requested
  m_events.WaitForResetAck(event_bits);
}

void MachProcess::SetExitInfo(const char *info) {
  if (info && info[0]) {
    DNBLogThreadedIf(LOG_PROCESS, "MachProcess::%s(\"%s\")", __FUNCTION__,
                     info);
    m_exit_info.assign(info);
  } else {
    DNBLogThreadedIf(LOG_PROCESS, "MachProcess::%s(NULL)", __FUNCTION__);
    m_exit_info.clear();
  }
}

void MachProcess::AppendSTDOUT(char *s, size_t len) {
  DNBLogThreadedIf(LOG_PROCESS, "MachProcess::%s (<%llu> %s) ...", __FUNCTION__,
                   (uint64_t)len, s);
  PTHREAD_MUTEX_LOCKER(locker, m_stdio_mutex);
  m_stdout_data.append(s, len);
  m_events.SetEvents(eEventStdioAvailable);

  // Wait for the event bit to reset if a reset ACK is requested
  m_events.WaitForResetAck(eEventStdioAvailable);
}

size_t MachProcess::GetAvailableSTDOUT(char *buf, size_t buf_size) {
  DNBLogThreadedIf(LOG_PROCESS, "MachProcess::%s (&%p[%llu]) ...", __FUNCTION__,
                   static_cast<void *>(buf), (uint64_t)buf_size);
  PTHREAD_MUTEX_LOCKER(locker, m_stdio_mutex);
  size_t bytes_available = m_stdout_data.size();
  if (bytes_available > 0) {
    if (bytes_available > buf_size) {
      memcpy(buf, m_stdout_data.data(), buf_size);
      m_stdout_data.erase(0, buf_size);
      bytes_available = buf_size;
    } else {
      memcpy(buf, m_stdout_data.data(), bytes_available);
      m_stdout_data.clear();
    }
  }
  return bytes_available;
}

nub_addr_t MachProcess::GetDYLDAllImageInfosAddress() {
  DNBError err;
  return m_task.GetDYLDAllImageInfosAddress(err);
}

size_t MachProcess::GetAvailableSTDERR(char *buf, size_t buf_size) { return 0; }

void *MachProcess::STDIOThread(void *arg) {
  MachProcess *proc = (MachProcess *)arg;
  DNBLogThreadedIf(LOG_PROCESS,
                   "MachProcess::%s ( arg = %p ) thread starting...",
                   __FUNCTION__, arg);

#if defined(__APPLE__)
  pthread_setname_np("stdio monitoring thread");
#endif

  // We start use a base and more options so we can control if we
  // are currently using a timeout on the mach_msg. We do this to get a
  // bunch of related exceptions on our exception port so we can process
  // then together. When we have multiple threads, we can get an exception
  // per thread and they will come in consecutively. The main thread loop
  // will start by calling mach_msg to without having the MACH_RCV_TIMEOUT
  // flag set in the options, so we will wait forever for an exception on
  // our exception port. After we get one exception, we then will use the
  // MACH_RCV_TIMEOUT option with a zero timeout to grab all other current
  // exceptions for our process. After we have received the last pending
  // exception, we will get a timeout which enables us to then notify
  // our main thread that we have an exception bundle available. We then wait
  // for the main thread to tell this exception thread to start trying to get
  // exceptions messages again and we start again with a mach_msg read with
  // infinite timeout.
  DNBError err;
  int stdout_fd = proc->GetStdoutFileDescriptor();
  int stderr_fd = proc->GetStderrFileDescriptor();
  if (stdout_fd == stderr_fd)
    stderr_fd = -1;

  while (stdout_fd >= 0 || stderr_fd >= 0) {
    ::pthread_testcancel();

    fd_set read_fds;
    FD_ZERO(&read_fds);
    if (stdout_fd >= 0)
      FD_SET(stdout_fd, &read_fds);
    if (stderr_fd >= 0)
      FD_SET(stderr_fd, &read_fds);
    int nfds = std::max<int>(stdout_fd, stderr_fd) + 1;

    int num_set_fds = select(nfds, &read_fds, NULL, NULL, NULL);
    DNBLogThreadedIf(LOG_PROCESS,
                     "select (nfds, &read_fds, NULL, NULL, NULL) => %d",
                     num_set_fds);

    if (num_set_fds < 0) {
      int select_errno = errno;
      if (DNBLogCheckLogBit(LOG_PROCESS)) {
        err.SetError(select_errno, DNBError::POSIX);
        err.LogThreadedIfError(
            "select (nfds, &read_fds, NULL, NULL, NULL) => %d", num_set_fds);
      }

      switch (select_errno) {
      case EAGAIN: // The kernel was (perhaps temporarily) unable to allocate
                   // the requested number of file descriptors, or we have
                   // non-blocking IO
        break;
      case EBADF: // One of the descriptor sets specified an invalid descriptor.
        return NULL;
        break;
      case EINTR:  // A signal was delivered before the time limit expired and
                   // before any of the selected events occurred.
      case EINVAL: // The specified time limit is invalid. One of its components
                   // is negative or too large.
      default:     // Other unknown error
        break;
      }
    } else if (num_set_fds == 0) {
    } else {
      char s[1024];
      s[sizeof(s) - 1] = '\0'; // Ensure we have NULL termination
      ssize_t bytes_read = 0;
      if (stdout_fd >= 0 && FD_ISSET(stdout_fd, &read_fds)) {
        do {
          bytes_read = ::read(stdout_fd, s, sizeof(s) - 1);
          if (bytes_read < 0) {
            int read_errno = errno;
            DNBLogThreadedIf(LOG_PROCESS,
                             "read (stdout_fd, ) => %zd   errno: %d (%s)",
                             bytes_read, read_errno, strerror(read_errno));
          } else if (bytes_read == 0) {
            // EOF...
            DNBLogThreadedIf(
                LOG_PROCESS,
                "read (stdout_fd, ) => %zd  (reached EOF for child STDOUT)",
                bytes_read);
            stdout_fd = -1;
          } else if (bytes_read > 0) {
            proc->AppendSTDOUT(s, bytes_read);
          }

        } while (bytes_read > 0);
      }

      if (stderr_fd >= 0 && FD_ISSET(stderr_fd, &read_fds)) {
        do {
          bytes_read = ::read(stderr_fd, s, sizeof(s) - 1);
          if (bytes_read < 0) {
            int read_errno = errno;
            DNBLogThreadedIf(LOG_PROCESS,
                             "read (stderr_fd, ) => %zd   errno: %d (%s)",
                             bytes_read, read_errno, strerror(read_errno));
          } else if (bytes_read == 0) {
            // EOF...
            DNBLogThreadedIf(
                LOG_PROCESS,
                "read (stderr_fd, ) => %zd  (reached EOF for child STDERR)",
                bytes_read);
            stderr_fd = -1;
          } else if (bytes_read > 0) {
            proc->AppendSTDOUT(s, bytes_read);
          }

        } while (bytes_read > 0);
      }
    }
  }
  DNBLogThreadedIf(LOG_PROCESS, "MachProcess::%s (%p): thread exiting...",
                   __FUNCTION__, arg);
  return NULL;
}

void MachProcess::SignalAsyncProfileData(const char *info) {
  DNBLogThreadedIf(LOG_PROCESS, "MachProcess::%s (%s) ...", __FUNCTION__, info);
  PTHREAD_MUTEX_LOCKER(locker, m_profile_data_mutex);
  m_profile_data.push_back(info);
  m_events.SetEvents(eEventProfileDataAvailable);

  // Wait for the event bit to reset if a reset ACK is requested
  m_events.WaitForResetAck(eEventProfileDataAvailable);
}

size_t MachProcess::GetAsyncProfileData(char *buf, size_t buf_size) {
  DNBLogThreadedIf(LOG_PROCESS, "MachProcess::%s (&%p[%llu]) ...", __FUNCTION__,
                   static_cast<void *>(buf), (uint64_t)buf_size);
  PTHREAD_MUTEX_LOCKER(locker, m_profile_data_mutex);
  if (m_profile_data.empty())
    return 0;

  size_t bytes_available = m_profile_data.front().size();
  if (bytes_available > 0) {
    if (bytes_available > buf_size) {
      memcpy(buf, m_profile_data.front().data(), buf_size);
      m_profile_data.front().erase(0, buf_size);
      bytes_available = buf_size;
    } else {
      memcpy(buf, m_profile_data.front().data(), bytes_available);
      m_profile_data.erase(m_profile_data.begin());
    }
  }
  return bytes_available;
}

void *MachProcess::ProfileThread(void *arg) {
  MachProcess *proc = (MachProcess *)arg;
  DNBLogThreadedIf(LOG_PROCESS,
                   "MachProcess::%s ( arg = %p ) thread starting...",
                   __FUNCTION__, arg);

#if defined(__APPLE__)
  pthread_setname_np("performance profiling thread");
#endif

  while (proc->IsProfilingEnabled()) {
    nub_state_t state = proc->GetState();
    if (state == eStateRunning) {
      std::string data =
          proc->Task().GetProfileData(proc->GetProfileScanType());
      if (!data.empty()) {
        proc->SignalAsyncProfileData(data.c_str());
      }
    } else if ((state == eStateUnloaded) || (state == eStateDetached) ||
               (state == eStateUnloaded)) {
      // Done. Get out of this thread.
      break;
    }
    timespec ts;
    {
      using namespace std::chrono;
      std::chrono::microseconds dur(proc->ProfileInterval());
      const auto dur_secs = duration_cast<seconds>(dur);
      const auto dur_usecs = dur % std::chrono::seconds(1);
      DNBTimer::OffsetTimeOfDay(&ts, dur_secs.count(), 
                                dur_usecs.count());
    }
    uint32_t bits_set = 
        proc->m_profile_events.WaitForSetEvents(eMachProcessProfileCancel, &ts);
    // If we got bits back, we were told to exit.  Do so.
    if (bits_set & eMachProcessProfileCancel)
      break;
  }
  return NULL;
}

pid_t MachProcess::AttachForDebug(pid_t pid, char *err_str, size_t err_len) {
  // Clear out and clean up from any current state
  Clear();
  if (pid != 0) {
    DNBError err;
    // Make sure the process exists...
    if (::getpgid(pid) < 0) {
      err.SetErrorToErrno();
      const char *err_cstr = err.AsString();
      ::snprintf(err_str, err_len, "%s",
                 err_cstr ? err_cstr : "No such process");
      DNBLogError ("MachProcess::AttachForDebug pid %d does not exist", pid);
      return INVALID_NUB_PROCESS;
    }

    SetState(eStateAttaching);
    m_pid = pid;
    if (!m_task.StartExceptionThread(err)) {
      const char *err_cstr = err.AsString();
      ::snprintf(err_str, err_len, "%s",
                 err_cstr ? err_cstr : "unable to start the exception thread");
      DNBLogThreadedIf(LOG_PROCESS, "error: failed to attach to pid %d", pid);
      DNBLogError ("MachProcess::AttachForDebug failed to start exception thread: %s", err_str);
      m_pid = INVALID_NUB_PROCESS;
      return INVALID_NUB_PROCESS;
    }

    errno = 0;
    if (::ptrace(PT_ATTACHEXC, pid, 0, 0)) {
      err.SetError(errno);
      DNBLogError ("MachProcess::AttachForDebug failed to ptrace(PT_ATTACHEXC): %s", err.AsString());
    } else {
      err.Clear();
    }

    if (err.Success()) {
      m_flags |= eMachProcessFlagsAttached;
      // Sleep a bit to let the exception get received and set our process
      // status
      // to stopped.
      ::usleep(250000);
      DNBLogThreadedIf(LOG_PROCESS, "successfully attached to pid %d", pid);
      return m_pid;
    } else {
      ::snprintf(err_str, err_len, "%s", err.AsString());
      DNBLogError ("MachProcess::AttachForDebug error: failed to attach to pid %d", pid);

      struct kinfo_proc kinfo;
      int mib[] = {CTL_KERN, KERN_PROC, KERN_PROC_PID, pid};
      size_t len = sizeof(struct kinfo_proc);
      if (sysctl(mib, sizeof(mib) / sizeof(mib[0]), &kinfo, &len, NULL, 0) == 0 && len > 0) {
        if (kinfo.kp_proc.p_flag & P_TRACED) {
          ::snprintf(err_str, err_len, "%s - process %d is already being debugged", err.AsString(), pid);
          DNBLogError ("MachProcess::AttachForDebug pid %d is already being debugged", pid);
        }
      }
    }
  }
  return INVALID_NUB_PROCESS;
}

Genealogy::ThreadActivitySP
MachProcess::GetGenealogyInfoForThread(nub_thread_t tid, bool &timed_out) {
  return m_activities.GetGenealogyInfoForThread(m_pid, tid, m_thread_list,
                                                m_task.TaskPort(), timed_out);
}

Genealogy::ProcessExecutableInfoSP
MachProcess::GetGenealogyImageInfo(size_t idx) {
  return m_activities.GetProcessExecutableInfosAtIndex(idx);
}

bool MachProcess::GetOSVersionNumbers(uint64_t *major, uint64_t *minor,
                                      uint64_t *patch) {
#if defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) &&                  \
    (__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ < 101000)
  return false;
#else
  NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

  NSOperatingSystemVersion vers =
      [[NSProcessInfo processInfo] operatingSystemVersion];
  if (major)
    *major = vers.majorVersion;
  if (minor)
    *minor = vers.minorVersion;
  if (patch)
    *patch = vers.patchVersion;

  [pool drain];

  return true;
#endif
}

std::string MachProcess::GetMacCatalystVersionString() {
  @autoreleasepool {
    NSDictionary *version_info =
      [NSDictionary dictionaryWithContentsOfFile:
       @"/System/Library/CoreServices/SystemVersion.plist"];
    NSString *version_value = [version_info objectForKey: @"iOSSupportVersion"];
    if (const char *version_str = [version_value UTF8String])
      return version_str;
  }
  return {};
}

// Do the process specific setup for attach.  If this returns NULL, then there's
// no
// platform specific stuff to be done to wait for the attach.  If you get
// non-null,
// pass that token to the CheckForProcess method, and then to
// CleanupAfterAttach.

//  Call PrepareForAttach before attaching to a process that has not yet
//  launched
// This returns a token that can be passed to CheckForProcess, and to
// CleanupAfterAttach.
// You should call CleanupAfterAttach to free the token, and do whatever other
// cleanup seems good.

const void *MachProcess::PrepareForAttach(const char *path,
                                          nub_launch_flavor_t launch_flavor,
                                          bool waitfor, DNBError &attach_err) {
#if defined(WITH_SPRINGBOARD) || defined(WITH_BKS) || defined(WITH_FBS)
  // Tell SpringBoard to halt the next launch of this application on startup.

  if (!waitfor)
    return NULL;

  const char *app_ext = strstr(path, ".app");
  const bool is_app =
      app_ext != NULL && (app_ext[4] == '\0' || app_ext[4] == '/');
  if (!is_app) {
    DNBLogThreadedIf(
        LOG_PROCESS,
        "MachProcess::PrepareForAttach(): path '%s' doesn't contain .app, "
        "we can't tell springboard to wait for launch...",
        path);
    return NULL;
  }

#if defined(WITH_FBS)
  if (launch_flavor == eLaunchFlavorDefault)
    launch_flavor = eLaunchFlavorFBS;
  if (launch_flavor != eLaunchFlavorFBS)
    return NULL;
#elif defined(WITH_BKS)
  if (launch_flavor == eLaunchFlavorDefault)
    launch_flavor = eLaunchFlavorBKS;
  if (launch_flavor != eLaunchFlavorBKS)
    return NULL;
#elif defined(WITH_SPRINGBOARD)
  if (launch_flavor == eLaunchFlavorDefault)
    launch_flavor = eLaunchFlavorSpringBoard;
  if (launch_flavor != eLaunchFlavorSpringBoard)
    return NULL;
#endif

  std::string app_bundle_path(path, app_ext + strlen(".app"));

  CFStringRef bundleIDCFStr =
      CopyBundleIDForPath(app_bundle_path.c_str(), attach_err);
  std::string bundleIDStr;
  CFString::UTF8(bundleIDCFStr, bundleIDStr);
  DNBLogThreadedIf(LOG_PROCESS,
                   "CopyBundleIDForPath (%s, err_str) returned @\"%s\"",
                   app_bundle_path.c_str(), bundleIDStr.c_str());

  if (bundleIDCFStr == NULL) {
    return NULL;
  }

#if defined(WITH_FBS)
  if (launch_flavor == eLaunchFlavorFBS) {
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

    NSString *stdio_path = nil;
    NSFileManager *file_manager = [NSFileManager defaultManager];
    const char *null_path = "/dev/null";
    stdio_path =
        [file_manager stringWithFileSystemRepresentation:null_path
                                                  length:strlen(null_path)];

    NSMutableDictionary *debug_options = [NSMutableDictionary dictionary];
    NSMutableDictionary *options = [NSMutableDictionary dictionary];

    DNBLogThreadedIf(LOG_PROCESS, "Calling BKSSystemService openApplication: "
                                  "@\"%s\",options include stdio path: \"%s\", "
                                  "BKSDebugOptionKeyDebugOnNextLaunch & "
                                  "BKSDebugOptionKeyWaitForDebugger )",
                     bundleIDStr.c_str(), null_path);

    [debug_options setObject:stdio_path
                      forKey:FBSDebugOptionKeyStandardOutPath];
    [debug_options setObject:stdio_path
                      forKey:FBSDebugOptionKeyStandardErrorPath];
    [debug_options setObject:[NSNumber numberWithBool:YES]
                      forKey:FBSDebugOptionKeyWaitForDebugger];
    [debug_options setObject:[NSNumber numberWithBool:YES]
                      forKey:FBSDebugOptionKeyDebugOnNextLaunch];

    [options setObject:debug_options
                forKey:FBSOpenApplicationOptionKeyDebuggingOptions];

    FBSSystemService *system_service = [[FBSSystemService alloc] init];

    mach_port_t client_port = [system_service createClientPort];
    __block dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
    __block FBSOpenApplicationErrorCode attach_error_code =
        FBSOpenApplicationErrorCodeNone;

    NSString *bundleIDNSStr = (NSString *)bundleIDCFStr;

    [system_service openApplication:bundleIDNSStr
                            options:options
                         clientPort:client_port
                         withResult:^(NSError *error) {
                           // The system service will cleanup the client port we
                           // created for us.
                           if (error)
                             attach_error_code =
                                 (FBSOpenApplicationErrorCode)[error code];

                           [system_service release];
                           dispatch_semaphore_signal(semaphore);
                         }];

    const uint32_t timeout_secs = 9;

    dispatch_time_t timeout =
        dispatch_time(DISPATCH_TIME_NOW, timeout_secs * NSEC_PER_SEC);

    long success = dispatch_semaphore_wait(semaphore, timeout) == 0;

    if (!success) {
      DNBLogError("timed out trying to launch %s.", bundleIDStr.c_str());
      attach_err.SetErrorString(
          "debugserver timed out waiting for openApplication to complete.");
      attach_err.SetError(OPEN_APPLICATION_TIMEOUT_ERROR, DNBError::Generic);
    } else if (attach_error_code != FBSOpenApplicationErrorCodeNone) {
      std::string empty_str;
      SetFBSError(attach_error_code, empty_str, attach_err);
      DNBLogError("unable to launch the application with CFBundleIdentifier "
                  "'%s' bks_error = %ld",
                  bundleIDStr.c_str(), (NSInteger)attach_error_code);
    }
    dispatch_release(semaphore);
    [pool drain];
  }
#endif
#if defined(WITH_BKS)
  if (launch_flavor == eLaunchFlavorBKS) {
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

    NSString *stdio_path = nil;
    NSFileManager *file_manager = [NSFileManager defaultManager];
    const char *null_path = "/dev/null";
    stdio_path =
        [file_manager stringWithFileSystemRepresentation:null_path
                                                  length:strlen(null_path)];

    NSMutableDictionary *debug_options = [NSMutableDictionary dictionary];
    NSMutableDictionary *options = [NSMutableDictionary dictionary];

    DNBLogThreadedIf(LOG_PROCESS, "Calling BKSSystemService openApplication: "
                                  "@\"%s\",options include stdio path: \"%s\", "
                                  "BKSDebugOptionKeyDebugOnNextLaunch & "
                                  "BKSDebugOptionKeyWaitForDebugger )",
                     bundleIDStr.c_str(), null_path);

    [debug_options setObject:stdio_path
                      forKey:BKSDebugOptionKeyStandardOutPath];
    [debug_options setObject:stdio_path
                      forKey:BKSDebugOptionKeyStandardErrorPath];
    [debug_options setObject:[NSNumber numberWithBool:YES]
                      forKey:BKSDebugOptionKeyWaitForDebugger];
    [debug_options setObject:[NSNumber numberWithBool:YES]
                      forKey:BKSDebugOptionKeyDebugOnNextLaunch];

    [options setObject:debug_options
                forKey:BKSOpenApplicationOptionKeyDebuggingOptions];

    BKSSystemService *system_service = [[BKSSystemService alloc] init];

    mach_port_t client_port = [system_service createClientPort];
    __block dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
    __block BKSOpenApplicationErrorCode attach_error_code =
        BKSOpenApplicationErrorCodeNone;

    NSString *bundleIDNSStr = (NSString *)bundleIDCFStr;

    [system_service openApplication:bundleIDNSStr
                            options:options
                         clientPort:client_port
                         withResult:^(NSError *error) {
                           // The system service will cleanup the client port we
                           // created for us.
                           if (error)
                             attach_error_code =
                                 (BKSOpenApplicationErrorCode)[error code];

                           [system_service release];
                           dispatch_semaphore_signal(semaphore);
                         }];

    const uint32_t timeout_secs = 9;

    dispatch_time_t timeout =
        dispatch_time(DISPATCH_TIME_NOW, timeout_secs * NSEC_PER_SEC);

    long success = dispatch_semaphore_wait(semaphore, timeout) == 0;

    if (!success) {
      DNBLogError("timed out trying to launch %s.", bundleIDStr.c_str());
      attach_err.SetErrorString(
          "debugserver timed out waiting for openApplication to complete.");
      attach_err.SetError(OPEN_APPLICATION_TIMEOUT_ERROR, DNBError::Generic);
    } else if (attach_error_code != BKSOpenApplicationErrorCodeNone) {
      std::string empty_str;
      SetBKSError(attach_error_code, empty_str, attach_err);
      DNBLogError("unable to launch the application with CFBundleIdentifier "
                  "'%s' bks_error = %ld",
                  bundleIDStr.c_str(), attach_error_code);
    }
    dispatch_release(semaphore);
    [pool drain];
  }
#endif

#if defined(WITH_SPRINGBOARD)
  if (launch_flavor == eLaunchFlavorSpringBoard) {
    SBSApplicationLaunchError sbs_error = 0;

    const char *stdout_err = "/dev/null";
    CFString stdio_path;
    stdio_path.SetFileSystemRepresentation(stdout_err);

    DNBLogThreadedIf(LOG_PROCESS, "SBSLaunchApplicationForDebugging ( @\"%s\" "
                                  ", NULL, NULL, NULL, @\"%s\", @\"%s\", "
                                  "SBSApplicationDebugOnNextLaunch | "
                                  "SBSApplicationLaunchWaitForDebugger )",
                     bundleIDStr.c_str(), stdout_err, stdout_err);

    sbs_error = SBSLaunchApplicationForDebugging(
        bundleIDCFStr,
        (CFURLRef)NULL, // openURL
        NULL,           // launch_argv.get(),
        NULL,           // launch_envp.get(),  // CFDictionaryRef environment
        stdio_path.get(), stdio_path.get(),
        SBSApplicationDebugOnNextLaunch | SBSApplicationLaunchWaitForDebugger);

    if (sbs_error != SBSApplicationLaunchErrorSuccess) {
      attach_err.SetError(sbs_error, DNBError::SpringBoard);
      return NULL;
    }
  }
#endif // WITH_SPRINGBOARD

  DNBLogThreadedIf(LOG_PROCESS, "Successfully set DebugOnNextLaunch.");
  return bundleIDCFStr;
#else // !(defined (WITH_SPRINGBOARD) || defined (WITH_BKS) || defined
      // (WITH_FBS))
  return NULL;
#endif
}

// Pass in the token you got from PrepareForAttach.  If there is a process
// for that token, then the pid will be returned, otherwise INVALID_NUB_PROCESS
// will be returned.

nub_process_t MachProcess::CheckForProcess(const void *attach_token,
                                           nub_launch_flavor_t launch_flavor) {
  if (attach_token == NULL)
    return INVALID_NUB_PROCESS;

#if defined(WITH_FBS)
  if (launch_flavor == eLaunchFlavorFBS) {
    NSString *bundleIDNSStr = (NSString *)attach_token;
    FBSSystemService *systemService = [[FBSSystemService alloc] init];
    pid_t pid = [systemService pidForApplication:bundleIDNSStr];
    [systemService release];
    if (pid == 0)
      return INVALID_NUB_PROCESS;
    else
      return pid;
  }
#endif

#if defined(WITH_BKS)
  if (launch_flavor == eLaunchFlavorBKS) {
    NSString *bundleIDNSStr = (NSString *)attach_token;
    BKSSystemService *systemService = [[BKSSystemService alloc] init];
    pid_t pid = [systemService pidForApplication:bundleIDNSStr];
    [systemService release];
    if (pid == 0)
      return INVALID_NUB_PROCESS;
    else
      return pid;
  }
#endif

#if defined(WITH_SPRINGBOARD)
  if (launch_flavor == eLaunchFlavorSpringBoard) {
    CFStringRef bundleIDCFStr = (CFStringRef)attach_token;
    Boolean got_it;
    nub_process_t attach_pid;
    got_it = SBSProcessIDForDisplayIdentifier(bundleIDCFStr, &attach_pid);
    if (got_it)
      return attach_pid;
    else
      return INVALID_NUB_PROCESS;
  }
#endif
  return INVALID_NUB_PROCESS;
}

// Call this to clean up after you have either attached or given up on the
// attach.
// Pass true for success if you have attached, false if you have not.
// The token will also be freed at this point, so you can't use it after calling
// this method.

void MachProcess::CleanupAfterAttach(const void *attach_token,
                                     nub_launch_flavor_t launch_flavor,
                                     bool success, DNBError &err_str) {
  if (attach_token == NULL)
    return;

#if defined(WITH_FBS)
  if (launch_flavor == eLaunchFlavorFBS) {
    if (!success) {
      FBSCleanupAfterAttach(attach_token, err_str);
    }
    CFRelease((CFStringRef)attach_token);
  }
#endif

#if defined(WITH_BKS)

  if (launch_flavor == eLaunchFlavorBKS) {
    if (!success) {
      BKSCleanupAfterAttach(attach_token, err_str);
    }
    CFRelease((CFStringRef)attach_token);
  }
#endif

#if defined(WITH_SPRINGBOARD)
  // Tell SpringBoard to cancel the debug on next launch of this application
  // if we failed to attach
  if (launch_flavor == eMachProcessFlagsUsingSpringBoard) {
    if (!success) {
      SBSApplicationLaunchError sbs_error = 0;
      CFStringRef bundleIDCFStr = (CFStringRef)attach_token;

      sbs_error = SBSLaunchApplicationForDebugging(
          bundleIDCFStr, (CFURLRef)NULL, NULL, NULL, NULL, NULL,
          SBSApplicationCancelDebugOnNextLaunch);

      if (sbs_error != SBSApplicationLaunchErrorSuccess) {
        err_str.SetError(sbs_error, DNBError::SpringBoard);
        return;
      }
    }

    CFRelease((CFStringRef)attach_token);
  }
#endif
}

pid_t MachProcess::LaunchForDebug(
    const char *path, char const *argv[], char const *envp[],
    const char *working_directory, // NULL => don't change, non-NULL => set
                                   // working directory for inferior to this
    const char *stdin_path, const char *stdout_path, const char *stderr_path,
    bool no_stdio, nub_launch_flavor_t launch_flavor, int disable_aslr,
    const char *event_data, DNBError &launch_err) {
  // Clear out and clean up from any current state
  Clear();

  DNBLogThreadedIf(LOG_PROCESS,
                   "%s( path = '%s', argv = %p, envp = %p, "
                   "launch_flavor = %u, disable_aslr = %d )",
                   __FUNCTION__, path, static_cast<const void *>(argv),
                   static_cast<const void *>(envp), launch_flavor,
                   disable_aslr);

  // Fork a child process for debugging
  SetState(eStateLaunching);

  switch (launch_flavor) {
  case eLaunchFlavorForkExec:
    m_pid = MachProcess::ForkChildForPTraceDebugging(path, argv, envp, this,
                                                     launch_err);
    break;
#ifdef WITH_FBS
  case eLaunchFlavorFBS: {
    const char *app_ext = strstr(path, ".app");
    if (app_ext && (app_ext[4] == '\0' || app_ext[4] == '/')) {
      std::string app_bundle_path(path, app_ext + strlen(".app"));
      m_flags |= (eMachProcessFlagsUsingFBS | eMachProcessFlagsBoardCalculated);
      if (BoardServiceLaunchForDebug(app_bundle_path.c_str(), argv, envp,
                                     no_stdio, disable_aslr, event_data,
                                     launch_err) != 0)
        return m_pid; // A successful SBLaunchForDebug() returns and assigns a
                      // non-zero m_pid.
      else
        break; // We tried a FBS launch, but didn't succeed lets get out
    }
  } break;
#endif
#ifdef WITH_BKS
  case eLaunchFlavorBKS: {
    const char *app_ext = strstr(path, ".app");
    if (app_ext && (app_ext[4] == '\0' || app_ext[4] == '/')) {
      std::string app_bundle_path(path, app_ext + strlen(".app"));
      m_flags |= (eMachProcessFlagsUsingBKS | eMachProcessFlagsBoardCalculated);
      if (BoardServiceLaunchForDebug(app_bundle_path.c_str(), argv, envp,
                                     no_stdio, disable_aslr, event_data,
                                     launch_err) != 0)
        return m_pid; // A successful SBLaunchForDebug() returns and assigns a
                      // non-zero m_pid.
      else
        break; // We tried a BKS launch, but didn't succeed lets get out
    }
  } break;
#endif
#ifdef WITH_SPRINGBOARD

  case eLaunchFlavorSpringBoard: {
    //  .../whatever.app/whatever ?
    //  Or .../com.apple.whatever.app/whatever -- be careful of ".app" in
    //  "com.apple.whatever" here
    const char *app_ext = strstr(path, ".app/");
    if (app_ext == NULL) {
      // .../whatever.app ?
      int len = strlen(path);
      if (len > 5) {
        if (strcmp(path + len - 4, ".app") == 0) {
          app_ext = path + len - 4;
        }
      }
    }
    if (app_ext) {
      std::string app_bundle_path(path, app_ext + strlen(".app"));
      if (SBLaunchForDebug(app_bundle_path.c_str(), argv, envp, no_stdio,
                           disable_aslr, launch_err) != 0)
        return m_pid; // A successful SBLaunchForDebug() returns and assigns a
                      // non-zero m_pid.
      else
        break; // We tried a springboard launch, but didn't succeed lets get out
    }
  } break;

#endif

  case eLaunchFlavorPosixSpawn:
    m_pid = MachProcess::PosixSpawnChildForPTraceDebugging(
        path, DNBArchProtocol::GetArchitecture(), argv, envp, working_directory,
        stdin_path, stdout_path, stderr_path, no_stdio, this, disable_aslr,
        launch_err);
    break;

  default:
    // Invalid  launch
    launch_err.SetError(NUB_GENERIC_ERROR, DNBError::Generic);
    return INVALID_NUB_PROCESS;
  }

  if (m_pid == INVALID_NUB_PROCESS) {
    // If we don't have a valid process ID and no one has set the error,
    // then return a generic error
    if (launch_err.Success())
      launch_err.SetError(NUB_GENERIC_ERROR, DNBError::Generic);
  } else {
    m_path = path;
    size_t i;
    char const *arg;
    for (i = 0; (arg = argv[i]) != NULL; i++)
      m_args.push_back(arg);

    m_task.StartExceptionThread(launch_err);
    if (launch_err.Fail()) {
      if (launch_err.AsString() == NULL)
        launch_err.SetErrorString("unable to start the exception thread");
      DNBLog("Could not get inferior's Mach exception port, sending ptrace "
             "PT_KILL and exiting.");
      ::ptrace(PT_KILL, m_pid, 0, 0);
      m_pid = INVALID_NUB_PROCESS;
      return INVALID_NUB_PROCESS;
    }

    StartSTDIOThread();

    if (launch_flavor == eLaunchFlavorPosixSpawn) {

      SetState(eStateAttaching);
      errno = 0;
      int err = ::ptrace(PT_ATTACHEXC, m_pid, 0, 0);
      if (err == 0) {
        m_flags |= eMachProcessFlagsAttached;
        DNBLogThreadedIf(LOG_PROCESS, "successfully spawned pid %d", m_pid);
        launch_err.Clear();
      } else {
        SetState(eStateExited);
        DNBError ptrace_err(errno, DNBError::POSIX);
        DNBLogThreadedIf(LOG_PROCESS, "error: failed to attach to spawned pid "
                                      "%d (err = %i, errno = %i (%s))",
                         m_pid, err, ptrace_err.Status(),
                         ptrace_err.AsString());
        launch_err.SetError(NUB_GENERIC_ERROR, DNBError::Generic);
      }
    } else {
      launch_err.Clear();
    }
  }
  return m_pid;
}

pid_t MachProcess::PosixSpawnChildForPTraceDebugging(
    const char *path, cpu_type_t cpu_type, char const *argv[],
    char const *envp[], const char *working_directory, const char *stdin_path,
    const char *stdout_path, const char *stderr_path, bool no_stdio,
    MachProcess *process, int disable_aslr, DNBError &err) {
  posix_spawnattr_t attr;
  short flags;
  DNBLogThreadedIf(LOG_PROCESS,
                   "%s ( path='%s', argv=%p, envp=%p, "
                   "working_dir=%s, stdin=%s, stdout=%s "
                   "stderr=%s, no-stdio=%i)",
                   __FUNCTION__, path, static_cast<const void *>(argv),
                   static_cast<const void *>(envp), working_directory,
                   stdin_path, stdout_path, stderr_path, no_stdio);

  err.SetError(::posix_spawnattr_init(&attr), DNBError::POSIX);
  if (err.Fail() || DNBLogCheckLogBit(LOG_PROCESS))
    err.LogThreaded("::posix_spawnattr_init ( &attr )");
  if (err.Fail())
    return INVALID_NUB_PROCESS;

  flags = POSIX_SPAWN_START_SUSPENDED | POSIX_SPAWN_SETSIGDEF |
          POSIX_SPAWN_SETSIGMASK;
  if (disable_aslr)
    flags |= _POSIX_SPAWN_DISABLE_ASLR;

  sigset_t no_signals;
  sigset_t all_signals;
  sigemptyset(&no_signals);
  sigfillset(&all_signals);
  ::posix_spawnattr_setsigmask(&attr, &no_signals);
  ::posix_spawnattr_setsigdefault(&attr, &all_signals);

  err.SetError(::posix_spawnattr_setflags(&attr, flags), DNBError::POSIX);
  if (err.Fail() || DNBLogCheckLogBit(LOG_PROCESS))
    err.LogThreaded(
        "::posix_spawnattr_setflags ( &attr, POSIX_SPAWN_START_SUSPENDED%s )",
        flags & _POSIX_SPAWN_DISABLE_ASLR ? " | _POSIX_SPAWN_DISABLE_ASLR"
                                          : "");
  if (err.Fail())
    return INVALID_NUB_PROCESS;

// Don't do this on SnowLeopard, _sometimes_ the TASK_BASIC_INFO will fail
// and we will fail to continue with our process...

// On SnowLeopard we should set "DYLD_NO_PIE" in the inferior environment....

#if !defined(__arm__)

  // We don't need to do this for ARM, and we really shouldn't now that we
  // have multiple CPU subtypes and no posix_spawnattr call that allows us
  // to set which CPU subtype to launch...
  if (cpu_type != 0) {
    size_t ocount = 0;
    err.SetError(::posix_spawnattr_setbinpref_np(&attr, 1, &cpu_type, &ocount),
                 DNBError::POSIX);
    if (err.Fail() || DNBLogCheckLogBit(LOG_PROCESS))
      err.LogThreaded("::posix_spawnattr_setbinpref_np ( &attr, 1, cpu_type = "
                      "0x%8.8x, count => %llu )",
                      cpu_type, (uint64_t)ocount);

    if (err.Fail() != 0 || ocount != 1)
      return INVALID_NUB_PROCESS;
  }
#endif

  PseudoTerminal pty;

  posix_spawn_file_actions_t file_actions;
  err.SetError(::posix_spawn_file_actions_init(&file_actions), DNBError::POSIX);
  int file_actions_valid = err.Success();
  if (!file_actions_valid || DNBLogCheckLogBit(LOG_PROCESS))
    err.LogThreaded("::posix_spawn_file_actions_init ( &file_actions )");
  int pty_error = -1;
  pid_t pid = INVALID_NUB_PROCESS;
  if (file_actions_valid) {
    if (stdin_path == NULL && stdout_path == NULL && stderr_path == NULL &&
        !no_stdio) {
      pty_error = pty.OpenFirstAvailablePrimary(O_RDWR | O_NOCTTY);
      if (pty_error == PseudoTerminal::success) {
        stdin_path = stdout_path = stderr_path = pty.SecondaryName();
      }
    }

    // if no_stdio or std paths not supplied, then route to "/dev/null".
    if (no_stdio || stdin_path == NULL || stdin_path[0] == '\0')
      stdin_path = "/dev/null";
    if (no_stdio || stdout_path == NULL || stdout_path[0] == '\0')
      stdout_path = "/dev/null";
    if (no_stdio || stderr_path == NULL || stderr_path[0] == '\0')
      stderr_path = "/dev/null";

    err.SetError(::posix_spawn_file_actions_addopen(&file_actions, STDIN_FILENO,
                                                    stdin_path,
                                                    O_RDONLY | O_NOCTTY, 0),
                 DNBError::POSIX);
    if (err.Fail() || DNBLogCheckLogBit(LOG_PROCESS))
      err.LogThreaded("::posix_spawn_file_actions_addopen (&file_actions, "
                      "filedes=STDIN_FILENO, path='%s')",
                      stdin_path);

    err.SetError(::posix_spawn_file_actions_addopen(
                     &file_actions, STDOUT_FILENO, stdout_path,
                     O_WRONLY | O_NOCTTY | O_CREAT, 0640),
                 DNBError::POSIX);
    if (err.Fail() || DNBLogCheckLogBit(LOG_PROCESS))
      err.LogThreaded("::posix_spawn_file_actions_addopen (&file_actions, "
                      "filedes=STDOUT_FILENO, path='%s')",
                      stdout_path);

    err.SetError(::posix_spawn_file_actions_addopen(
                     &file_actions, STDERR_FILENO, stderr_path,
                     O_WRONLY | O_NOCTTY | O_CREAT, 0640),
                 DNBError::POSIX);
    if (err.Fail() || DNBLogCheckLogBit(LOG_PROCESS))
      err.LogThreaded("::posix_spawn_file_actions_addopen (&file_actions, "
                      "filedes=STDERR_FILENO, path='%s')",
                      stderr_path);

    // TODO: Verify if we can set the working directory back immediately
    // after the posix_spawnp call without creating a race condition???
    if (working_directory)
      ::chdir(working_directory);

    err.SetError(::posix_spawnp(&pid, path, &file_actions, &attr,
                                const_cast<char *const *>(argv),
                                const_cast<char *const *>(envp)),
                 DNBError::POSIX);
    if (err.Fail() || DNBLogCheckLogBit(LOG_PROCESS))
      err.LogThreaded("::posix_spawnp ( pid => %i, path = '%s', file_actions = "
                      "%p, attr = %p, argv = %p, envp = %p )",
                      pid, path, &file_actions, &attr, argv, envp);
  } else {
    // TODO: Verify if we can set the working directory back immediately
    // after the posix_spawnp call without creating a race condition???
    if (working_directory)
      ::chdir(working_directory);

    err.SetError(::posix_spawnp(&pid, path, NULL, &attr,
                                const_cast<char *const *>(argv),
                                const_cast<char *const *>(envp)),
                 DNBError::POSIX);
    if (err.Fail() || DNBLogCheckLogBit(LOG_PROCESS))
      err.LogThreaded("::posix_spawnp ( pid => %i, path = '%s', file_actions = "
                      "%p, attr = %p, argv = %p, envp = %p )",
                      pid, path, NULL, &attr, argv, envp);
  }

  // We have seen some cases where posix_spawnp was returning a valid
  // looking pid even when an error was returned, so clear it out
  if (err.Fail())
    pid = INVALID_NUB_PROCESS;

  if (pty_error == 0) {
    if (process != NULL) {
      int primary_fd = pty.ReleasePrimaryFD();
      process->SetChildFileDescriptors(primary_fd, primary_fd, primary_fd);
    }
  }
  ::posix_spawnattr_destroy(&attr);

  if (pid != INVALID_NUB_PROCESS) {
    cpu_type_t pid_cpu_type = MachProcess::GetCPUTypeForLocalProcess(pid);
    DNBLogThreadedIf(LOG_PROCESS,
                     "MachProcess::%s ( ) pid=%i, cpu_type=0x%8.8x",
                     __FUNCTION__, pid, pid_cpu_type);
    if (pid_cpu_type)
      DNBArchProtocol::SetArchitecture(pid_cpu_type);
  }

  if (file_actions_valid) {
    DNBError err2;
    err2.SetError(::posix_spawn_file_actions_destroy(&file_actions),
                  DNBError::POSIX);
    if (err2.Fail() || DNBLogCheckLogBit(LOG_PROCESS))
      err2.LogThreaded("::posix_spawn_file_actions_destroy ( &file_actions )");
  }

  return pid;
}

uint32_t MachProcess::GetCPUTypeForLocalProcess(pid_t pid) {
  int mib[CTL_MAXNAME] = {
      0,
  };
  size_t len = CTL_MAXNAME;
  if (::sysctlnametomib("sysctl.proc_cputype", mib, &len))
    return 0;

  mib[len] = pid;
  len++;

  cpu_type_t cpu;
  size_t cpu_len = sizeof(cpu);
  if (::sysctl(mib, static_cast<u_int>(len), &cpu, &cpu_len, 0, 0))
    cpu = 0;
  return cpu;
}

pid_t MachProcess::ForkChildForPTraceDebugging(const char *path,
                                               char const *argv[],
                                               char const *envp[],
                                               MachProcess *process,
                                               DNBError &launch_err) {
  PseudoTerminal::Status pty_error = PseudoTerminal::success;

  // Use a fork that ties the child process's stdin/out/err to a pseudo
  // terminal so we can read it in our MachProcess::STDIOThread
  // as unbuffered io.
  PseudoTerminal pty;
  pid_t pid = pty.Fork(pty_error);

  if (pid < 0) {
    //--------------------------------------------------------------
    // Status during fork.
    //--------------------------------------------------------------
    return pid;
  } else if (pid == 0) {
    //--------------------------------------------------------------
    // Child process
    //--------------------------------------------------------------
    ::ptrace(PT_TRACE_ME, 0, 0, 0); // Debug this process
    ::ptrace(PT_SIGEXC, 0, 0, 0);   // Get BSD signals as mach exceptions

    // If our parent is setgid, lets make sure we don't inherit those
    // extra powers due to nepotism.
    if (::setgid(getgid()) == 0) {

      // Let the child have its own process group. We need to execute
      // this call in both the child and parent to avoid a race condition
      // between the two processes.
      ::setpgid(0, 0); // Set the child process group to match its pid

      // Sleep a bit to before the exec call
      ::sleep(1);

      // Turn this process into
      ::execv(path, const_cast<char *const *>(argv));
    }
    // Exit with error code. Child process should have taken
    // over in above exec call and if the exec fails it will
    // exit the child process below.
    ::exit(127);
  } else {
    //--------------------------------------------------------------
    // Parent process
    //--------------------------------------------------------------
    // Let the child have its own process group. We need to execute
    // this call in both the child and parent to avoid a race condition
    // between the two processes.
    ::setpgid(pid, pid); // Set the child process group to match its pid

    if (process != NULL) {
      // Release our primary pty file descriptor so the pty class doesn't
      // close it and so we can continue to use it in our STDIO thread
      int primary_fd = pty.ReleasePrimaryFD();
      process->SetChildFileDescriptors(primary_fd, primary_fd, primary_fd);
    }
  }
  return pid;
}

#if defined(WITH_SPRINGBOARD) || defined(WITH_BKS) || defined(WITH_FBS)
// This returns a CFRetained pointer to the Bundle ID for app_bundle_path,
// or NULL if there was some problem getting the bundle id.
static CFStringRef CopyBundleIDForPath(const char *app_bundle_path,
                                       DNBError &err_str) {
  CFBundle bundle(app_bundle_path);
  CFStringRef bundleIDCFStr = bundle.GetIdentifier();
  std::string bundleID;
  if (CFString::UTF8(bundleIDCFStr, bundleID) == NULL) {
    struct stat app_bundle_stat;
    char err_msg[PATH_MAX];

    if (::stat(app_bundle_path, &app_bundle_stat) < 0) {
      err_str.SetError(errno, DNBError::POSIX);
      snprintf(err_msg, sizeof(err_msg), "%s: \"%s\"", err_str.AsString(),
               app_bundle_path);
      err_str.SetErrorString(err_msg);
      DNBLogThreadedIf(LOG_PROCESS, "%s() error: %s", __FUNCTION__, err_msg);
    } else {
      err_str.SetError(-1, DNBError::Generic);
      snprintf(err_msg, sizeof(err_msg),
               "failed to extract CFBundleIdentifier from %s", app_bundle_path);
      err_str.SetErrorString(err_msg);
      DNBLogThreadedIf(
          LOG_PROCESS,
          "%s() error: failed to extract CFBundleIdentifier from '%s'",
          __FUNCTION__, app_bundle_path);
    }
    return NULL;
  }

  DNBLogThreadedIf(LOG_PROCESS, "%s() extracted CFBundleIdentifier: %s",
                   __FUNCTION__, bundleID.c_str());
  CFRetain(bundleIDCFStr);

  return bundleIDCFStr;
}
#endif // #if defined (WITH_SPRINGBOARD) || defined (WITH_BKS) || defined
       // (WITH_FBS)
#ifdef WITH_SPRINGBOARD

pid_t MachProcess::SBLaunchForDebug(const char *path, char const *argv[],
                                    char const *envp[], bool no_stdio,
                                    bool disable_aslr, DNBError &launch_err) {
  // Clear out and clean up from any current state
  Clear();

  DNBLogThreadedIf(LOG_PROCESS, "%s( '%s', argv)", __FUNCTION__, path);

  // Fork a child process for debugging
  SetState(eStateLaunching);
  m_pid = MachProcess::SBForkChildForPTraceDebugging(path, argv, envp, no_stdio,
                                                     this, launch_err);
  if (m_pid != 0) {
    m_path = path;
    size_t i;
    char const *arg;
    for (i = 0; (arg = argv[i]) != NULL; i++)
      m_args.push_back(arg);
    m_task.StartExceptionThread(launch_err);

    if (launch_err.Fail()) {
      if (launch_err.AsString() == NULL)
        launch_err.SetErrorString("unable to start the exception thread");
      DNBLog("Could not get inferior's Mach exception port, sending ptrace "
             "PT_KILL and exiting.");
      ::ptrace(PT_KILL, m_pid, 0, 0);
      m_pid = INVALID_NUB_PROCESS;
      return INVALID_NUB_PROCESS;
    }

    StartSTDIOThread();
    SetState(eStateAttaching);
    int err = ::ptrace(PT_ATTACHEXC, m_pid, 0, 0);
    if (err == 0) {
      m_flags |= eMachProcessFlagsAttached;
      DNBLogThreadedIf(LOG_PROCESS, "successfully attached to pid %d", m_pid);
    } else {
      SetState(eStateExited);
      DNBLogThreadedIf(LOG_PROCESS, "error: failed to attach to pid %d", m_pid);
    }
  }
  return m_pid;
}

#include <servers/bootstrap.h>

pid_t MachProcess::SBForkChildForPTraceDebugging(
    const char *app_bundle_path, char const *argv[], char const *envp[],
    bool no_stdio, MachProcess *process, DNBError &launch_err) {
  DNBLogThreadedIf(LOG_PROCESS, "%s( '%s', argv, %p)", __FUNCTION__,
                   app_bundle_path, process);
  CFAllocatorRef alloc = kCFAllocatorDefault;

  if (argv[0] == NULL)
    return INVALID_NUB_PROCESS;

  size_t argc = 0;
  // Count the number of arguments
  while (argv[argc] != NULL)
    argc++;

  // Enumerate the arguments
  size_t first_launch_arg_idx = 1;
  CFReleaser<CFMutableArrayRef> launch_argv;

  if (argv[first_launch_arg_idx]) {
    size_t launch_argc = argc > 0 ? argc - 1 : 0;
    launch_argv.reset(
        ::CFArrayCreateMutable(alloc, launch_argc, &kCFTypeArrayCallBacks));
    size_t i;
    char const *arg;
    CFString launch_arg;
    for (i = first_launch_arg_idx; (i < argc) && ((arg = argv[i]) != NULL);
         i++) {
      launch_arg.reset(
          ::CFStringCreateWithCString(alloc, arg, kCFStringEncodingUTF8));
      if (launch_arg.get() != NULL)
        CFArrayAppendValue(launch_argv.get(), launch_arg.get());
      else
        break;
    }
  }

  // Next fill in the arguments dictionary.  Note, the envp array is of the form
  // Variable=value but SpringBoard wants a CF dictionary.  So we have to
  // convert
  // this here.

  CFReleaser<CFMutableDictionaryRef> launch_envp;

  if (envp[0]) {
    launch_envp.reset(
        ::CFDictionaryCreateMutable(alloc, 0, &kCFTypeDictionaryKeyCallBacks,
                                    &kCFTypeDictionaryValueCallBacks));
    const char *value;
    int name_len;
    CFString name_string, value_string;

    for (int i = 0; envp[i] != NULL; i++) {
      value = strstr(envp[i], "=");

      // If the name field is empty or there's no =, skip it.  Somebody's
      // messing with us.
      if (value == NULL || value == envp[i])
        continue;

      name_len = value - envp[i];

      // Now move value over the "="
      value++;

      name_string.reset(
          ::CFStringCreateWithBytes(alloc, (const UInt8 *)envp[i], name_len,
                                    kCFStringEncodingUTF8, false));
      value_string.reset(
          ::CFStringCreateWithCString(alloc, value, kCFStringEncodingUTF8));
      CFDictionarySetValue(launch_envp.get(), name_string.get(),
                           value_string.get());
    }
  }

  CFString stdio_path;

  PseudoTerminal pty;
  if (!no_stdio) {
    PseudoTerminal::Status pty_err =
        pty.OpenFirstAvailablePrimary(O_RDWR | O_NOCTTY);
    if (pty_err == PseudoTerminal::success) {
      const char *secondary_name = pty.SecondaryName();
      DNBLogThreadedIf(LOG_PROCESS,
                       "%s() successfully opened primary pty, secondary is %s",
                       __FUNCTION__, secondary_name);
      if (secondary_name && secondary_name[0]) {
        ::chmod(secondary_name, S_IRWXU | S_IRWXG | S_IRWXO);
        stdio_path.SetFileSystemRepresentation(secondary_name);
      }
    }
  }

  if (stdio_path.get() == NULL) {
    stdio_path.SetFileSystemRepresentation("/dev/null");
  }

  CFStringRef bundleIDCFStr = CopyBundleIDForPath(app_bundle_path, launch_err);
  if (bundleIDCFStr == NULL)
    return INVALID_NUB_PROCESS;

  // This is just for logging:
  std::string bundleID;
  CFString::UTF8(bundleIDCFStr, bundleID);

  DNBLogThreadedIf(LOG_PROCESS, "%s() serialized launch arg array",
                   __FUNCTION__);

  // Find SpringBoard
  SBSApplicationLaunchError sbs_error = 0;
  sbs_error = SBSLaunchApplicationForDebugging(
      bundleIDCFStr,
      (CFURLRef)NULL, // openURL
      launch_argv.get(),
      launch_envp.get(), // CFDictionaryRef environment
      stdio_path.get(), stdio_path.get(),
      SBSApplicationLaunchWaitForDebugger | SBSApplicationLaunchUnlockDevice);

  launch_err.SetError(sbs_error, DNBError::SpringBoard);

  if (sbs_error == SBSApplicationLaunchErrorSuccess) {
    static const useconds_t pid_poll_interval = 200000;
    static const useconds_t pid_poll_timeout = 30000000;

    useconds_t pid_poll_total = 0;

    nub_process_t pid = INVALID_NUB_PROCESS;
    Boolean pid_found = SBSProcessIDForDisplayIdentifier(bundleIDCFStr, &pid);
    // Poll until the process is running, as long as we are getting valid
    // responses and the timeout hasn't expired
    // A return PID of 0 means the process is not running, which may be because
    // it hasn't been (asynchronously) started
    // yet, or that it died very quickly (if you weren't using waitForDebugger).
    while (!pid_found && pid_poll_total < pid_poll_timeout) {
      usleep(pid_poll_interval);
      pid_poll_total += pid_poll_interval;
      DNBLogThreadedIf(LOG_PROCESS,
                       "%s() polling Springboard for pid for %s...",
                       __FUNCTION__, bundleID.c_str());
      pid_found = SBSProcessIDForDisplayIdentifier(bundleIDCFStr, &pid);
    }

    CFRelease(bundleIDCFStr);
    if (pid_found) {
      if (process != NULL) {
        // Release our primary pty file descriptor so the pty class doesn't
        // close it and so we can continue to use it in our STDIO thread
        int primary_fd = pty.ReleasePrimaryFD();
        process->SetChildFileDescriptors(primary_fd, primary_fd, primary_fd);
      }
      DNBLogThreadedIf(LOG_PROCESS, "%s() => pid = %4.4x", __FUNCTION__, pid);
    } else {
      DNBLogError("failed to lookup the process ID for CFBundleIdentifier %s.",
                  bundleID.c_str());
    }
    return pid;
  }

  DNBLogError("unable to launch the application with CFBundleIdentifier '%s' "
              "sbs_error = %u",
              bundleID.c_str(), sbs_error);
  return INVALID_NUB_PROCESS;
}

#endif // #ifdef WITH_SPRINGBOARD

#if defined(WITH_BKS) || defined(WITH_FBS)
pid_t MachProcess::BoardServiceLaunchForDebug(
    const char *path, char const *argv[], char const *envp[], bool no_stdio,
    bool disable_aslr, const char *event_data, DNBError &launch_err) {
  DNBLogThreadedIf(LOG_PROCESS, "%s( '%s', argv)", __FUNCTION__, path);

  // Fork a child process for debugging
  SetState(eStateLaunching);
  m_pid = BoardServiceForkChildForPTraceDebugging(
      path, argv, envp, no_stdio, disable_aslr, event_data, launch_err);
  if (m_pid != 0) {
    m_path = path;
    size_t i;
    char const *arg;
    for (i = 0; (arg = argv[i]) != NULL; i++)
      m_args.push_back(arg);
    m_task.StartExceptionThread(launch_err);

    if (launch_err.Fail()) {
      if (launch_err.AsString() == NULL)
        launch_err.SetErrorString("unable to start the exception thread");
      DNBLog("Could not get inferior's Mach exception port, sending ptrace "
             "PT_KILL and exiting.");
      ::ptrace(PT_KILL, m_pid, 0, 0);
      m_pid = INVALID_NUB_PROCESS;
      return INVALID_NUB_PROCESS;
    }

    StartSTDIOThread();
    SetState(eStateAttaching);
    int err = ::ptrace(PT_ATTACHEXC, m_pid, 0, 0);
    if (err == 0) {
      m_flags |= eMachProcessFlagsAttached;
      DNBLogThreadedIf(LOG_PROCESS, "successfully attached to pid %d", m_pid);
    } else {
      SetState(eStateExited);
      DNBLogThreadedIf(LOG_PROCESS, "error: failed to attach to pid %d", m_pid);
    }
  }
  return m_pid;
}

pid_t MachProcess::BoardServiceForkChildForPTraceDebugging(
    const char *app_bundle_path, char const *argv[], char const *envp[],
    bool no_stdio, bool disable_aslr, const char *event_data,
    DNBError &launch_err) {
  if (argv[0] == NULL)
    return INVALID_NUB_PROCESS;

  DNBLogThreadedIf(LOG_PROCESS, "%s( '%s', argv, %p)", __FUNCTION__,
                   app_bundle_path, this);

  NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

  size_t argc = 0;
  // Count the number of arguments
  while (argv[argc] != NULL)
    argc++;

  // Enumerate the arguments
  size_t first_launch_arg_idx = 1;

  NSMutableArray *launch_argv = nil;

  if (argv[first_launch_arg_idx]) {
    size_t launch_argc = argc > 0 ? argc - 1 : 0;
    launch_argv = [NSMutableArray arrayWithCapacity:launch_argc];
    size_t i;
    char const *arg;
    NSString *launch_arg;
    for (i = first_launch_arg_idx; (i < argc) && ((arg = argv[i]) != NULL);
         i++) {
      launch_arg = [NSString stringWithUTF8String:arg];
      // FIXME: Should we silently eat an argument that we can't convert into a
      // UTF8 string?
      if (launch_arg != nil)
        [launch_argv addObject:launch_arg];
      else
        break;
    }
  }

  NSMutableDictionary *launch_envp = nil;
  if (envp[0]) {
    launch_envp = [[NSMutableDictionary alloc] init];
    const char *value;
    int name_len;
    NSString *name_string, *value_string;

    for (int i = 0; envp[i] != NULL; i++) {
      value = strstr(envp[i], "=");

      // If the name field is empty or there's no =, skip it.  Somebody's
      // messing with us.
      if (value == NULL || value == envp[i])
        continue;

      name_len = value - envp[i];

      // Now move value over the "="
      value++;
      name_string = [[NSString alloc] initWithBytes:envp[i]
                                             length:name_len
                                           encoding:NSUTF8StringEncoding];
      value_string = [NSString stringWithUTF8String:value];
      [launch_envp setObject:value_string forKey:name_string];
    }
  }

  NSString *stdio_path = nil;
  NSFileManager *file_manager = [NSFileManager defaultManager];

  PseudoTerminal pty;
  if (!no_stdio) {
    PseudoTerminal::Status pty_err =
        pty.OpenFirstAvailablePrimary(O_RDWR | O_NOCTTY);
    if (pty_err == PseudoTerminal::success) {
      const char *secondary_name = pty.SecondaryName();
      DNBLogThreadedIf(LOG_PROCESS,
                       "%s() successfully opened primary pty, secondary is %s",
                       __FUNCTION__, secondary_name);
      if (secondary_name && secondary_name[0]) {
        ::chmod(secondary_name, S_IRWXU | S_IRWXG | S_IRWXO);
        stdio_path = [file_manager
            stringWithFileSystemRepresentation:secondary_name
                                        length:strlen(secondary_name)];
      }
    }
  }

  if (stdio_path == nil) {
    const char *null_path = "/dev/null";
    stdio_path =
        [file_manager stringWithFileSystemRepresentation:null_path
                                                  length:strlen(null_path)];
  }

  CFStringRef bundleIDCFStr = CopyBundleIDForPath(app_bundle_path, launch_err);
  if (bundleIDCFStr == NULL) {
    [pool drain];
    return INVALID_NUB_PROCESS;
  }

  // Instead of rewriting CopyBundleIDForPath for NSStrings, we'll just use
  // toll-free bridging here:
  NSString *bundleIDNSStr = (NSString *)bundleIDCFStr;

  // Okay, now let's assemble all these goodies into the BackBoardServices
  // options mega-dictionary:

  NSMutableDictionary *options = nullptr;
  pid_t return_pid = INVALID_NUB_PROCESS;
  bool success = false;

#ifdef WITH_BKS
  if (ProcessUsingBackBoard()) {
    options =
        BKSCreateOptionsDictionary(app_bundle_path, launch_argv, launch_envp,
                                   stdio_path, disable_aslr, event_data);
    success = BKSCallOpenApplicationFunction(bundleIDNSStr, options, launch_err,
                                             &return_pid);
  }
#endif
#ifdef WITH_FBS
  if (ProcessUsingFrontBoard()) {
    options =
        FBSCreateOptionsDictionary(app_bundle_path, launch_argv, launch_envp,
                                   stdio_path, disable_aslr, event_data);
    success = FBSCallOpenApplicationFunction(bundleIDNSStr, options, launch_err,
                                             &return_pid);
  }
#endif

  if (success) {
    int primary_fd = pty.ReleasePrimaryFD();
    SetChildFileDescriptors(primary_fd, primary_fd, primary_fd);
    CFString::UTF8(bundleIDCFStr, m_bundle_id);
  }

  [pool drain];

  return return_pid;
}

bool MachProcess::BoardServiceSendEvent(const char *event_data,
                                        DNBError &send_err) {
  bool return_value = true;

  if (event_data == NULL || *event_data == '\0') {
    DNBLogError("SendEvent called with NULL event data.");
    send_err.SetErrorString("SendEvent called with empty event data");
    return false;
  }

  NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

  if (strcmp(event_data, "BackgroundApplication") == 0) {
// This is an event I cooked up.  What you actually do is foreground the system
// app, so:
#ifdef WITH_BKS
    if (ProcessUsingBackBoard()) {
      return_value = BKSCallOpenApplicationFunction(nil, nil, send_err, NULL);
    }
#endif
#ifdef WITH_FBS
    if (ProcessUsingFrontBoard()) {
      return_value = FBSCallOpenApplicationFunction(nil, nil, send_err, NULL);
    }
#endif
    if (!return_value) {
      DNBLogError("Failed to background application, error: %s.",
                  send_err.AsString());
    }
  } else {
    if (m_bundle_id.empty()) {
      // See if we can figure out the bundle ID for this PID:

      DNBLogError(
          "Tried to send event \"%s\" to a process that has no bundle ID.",
          event_data);
      return false;
    }

    NSString *bundleIDNSStr =
        [NSString stringWithUTF8String:m_bundle_id.c_str()];

    NSMutableDictionary *options = [NSMutableDictionary dictionary];

#ifdef WITH_BKS
    if (ProcessUsingBackBoard()) {
      if (!BKSAddEventDataToOptions(options, event_data, send_err)) {
        [pool drain];
        return false;
      }
      return_value = BKSCallOpenApplicationFunction(bundleIDNSStr, options,
                                                    send_err, NULL);
      DNBLogThreadedIf(LOG_PROCESS,
                       "Called BKSCallOpenApplicationFunction to send event.");
    }
#endif
#ifdef WITH_FBS
    if (ProcessUsingFrontBoard()) {
      if (!FBSAddEventDataToOptions(options, event_data, send_err)) {
        [pool drain];
        return false;
      }
      return_value = FBSCallOpenApplicationFunction(bundleIDNSStr, options,
                                                    send_err, NULL);
      DNBLogThreadedIf(LOG_PROCESS,
                       "Called FBSCallOpenApplicationFunction to send event.");
    }
#endif

    if (!return_value) {
      DNBLogError("Failed to send event: %s, error: %s.", event_data,
                  send_err.AsString());
    }
  }

  [pool drain];
  return return_value;
}
#endif // defined(WITH_BKS) || defined (WITH_FBS)

#ifdef WITH_BKS
void MachProcess::BKSCleanupAfterAttach(const void *attach_token,
                                        DNBError &err_str) {
  bool success;

  NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

  // Instead of rewriting CopyBundleIDForPath for NSStrings, we'll just use
  // toll-free bridging here:
  NSString *bundleIDNSStr = (NSString *)attach_token;

  // Okay, now let's assemble all these goodies into the BackBoardServices
  // options mega-dictionary:

  // First we have the debug sub-dictionary:
  NSMutableDictionary *debug_options = [NSMutableDictionary dictionary];
  [debug_options setObject:[NSNumber numberWithBool:YES]
                    forKey:BKSDebugOptionKeyCancelDebugOnNextLaunch];

  // That will go in the overall dictionary:

  NSMutableDictionary *options = [NSMutableDictionary dictionary];
  [options setObject:debug_options
              forKey:BKSOpenApplicationOptionKeyDebuggingOptions];

  success =
      BKSCallOpenApplicationFunction(bundleIDNSStr, options, err_str, NULL);

  if (!success) {
    DNBLogError("error trying to cancel debug on next launch for %s: %s",
                [bundleIDNSStr UTF8String], err_str.AsString());
  }

  [pool drain];
}
#endif // WITH_BKS

#ifdef WITH_FBS
void MachProcess::FBSCleanupAfterAttach(const void *attach_token,
                                        DNBError &err_str) {
  bool success;

  NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

  // Instead of rewriting CopyBundleIDForPath for NSStrings, we'll just use
  // toll-free bridging here:
  NSString *bundleIDNSStr = (NSString *)attach_token;

  // Okay, now let's assemble all these goodies into the BackBoardServices
  // options mega-dictionary:

  // First we have the debug sub-dictionary:
  NSMutableDictionary *debug_options = [NSMutableDictionary dictionary];
  [debug_options setObject:[NSNumber numberWithBool:YES]
                    forKey:FBSDebugOptionKeyCancelDebugOnNextLaunch];

  // That will go in the overall dictionary:

  NSMutableDictionary *options = [NSMutableDictionary dictionary];
  [options setObject:debug_options
              forKey:FBSOpenApplicationOptionKeyDebuggingOptions];

  success =
      FBSCallOpenApplicationFunction(bundleIDNSStr, options, err_str, NULL);

  if (!success) {
    DNBLogError("error trying to cancel debug on next launch for %s: %s",
                [bundleIDNSStr UTF8String], err_str.AsString());
  }

  [pool drain];
}
#endif // WITH_FBS


void MachProcess::CalculateBoardStatus()
{
  if (m_flags & eMachProcessFlagsBoardCalculated)
    return;
  if (m_pid == 0)
    return;

#if defined (WITH_FBS) || defined (WITH_BKS)
    bool found_app_flavor = false;
#endif

#if defined(WITH_FBS)
    if (!found_app_flavor && IsFBSProcess(m_pid)) {
      found_app_flavor = true;
      m_flags |= eMachProcessFlagsUsingFBS;
    }
#endif
#if defined(WITH_BKS)
    if (!found_app_flavor && IsBKSProcess(m_pid)) {
      found_app_flavor = true;
      m_flags |= eMachProcessFlagsUsingBKS;
    }
#endif

    m_flags |= eMachProcessFlagsBoardCalculated;
}

bool MachProcess::ProcessUsingBackBoard() {
  CalculateBoardStatus();
  return (m_flags & eMachProcessFlagsUsingBKS) != 0;
}

bool MachProcess::ProcessUsingFrontBoard() {
  CalculateBoardStatus();
  return (m_flags & eMachProcessFlagsUsingFBS) != 0;
}
