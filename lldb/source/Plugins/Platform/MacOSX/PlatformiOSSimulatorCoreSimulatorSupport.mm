//===-- PlatformiOSSimulatorCoreSimulatorSupport.cpp ---------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PlatformiOSSimulatorCoreSimulatorSupport.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
#include <CoreFoundation/CoreFoundation.h>
#include <Foundation/Foundation.h>
// Project includes
#include "lldb/Host/PseudoTerminal.h"
#include "lldb/Target/FileAction.h"

#include "llvm/ADT/StringRef.h"

using namespace lldb_private;
using namespace lldb_utility;
// CoreSimulator lives as part of Xcode, which means we can't really link
// against it, so we dlopen()
// it at runtime, and error out nicely if that fails
@interface SimServiceContext {
}
+ (id)sharedServiceContextForDeveloperDir:(NSString *)dir
                                    error:(NSError **)error;
@end
// However, the drawback is that the compiler will not know about the selectors
// we're trying to use
// until runtime; to appease clang in this regard, define a fake protocol on
// NSObject that exposes
// the needed interface names for us
@protocol LLDBCoreSimulatorSupport <NSObject>
- (id)defaultDeviceSetWithError:(NSError **)error;
- (NSArray *)devices;
- (id)deviceType;
- (NSString *)name;
- (NSString *)identifier;
- (NSString *)modelIdentifier;
- (NSString *)productFamily;
- (int32_t)productFamilyID;
- (id)runtime;
- (BOOL)available;
- (NSString *)versionString;
- (NSString *)buildVersionString;
- (BOOL)bootWithOptions:(NSDictionary *)options error:(NSError **)error;
- (NSUInteger)state;
- (BOOL)shutdownWithError:(NSError **)error;
- (NSUUID *)UDID;
- (pid_t)spawnWithPath:(NSString *)path
               options:(NSDictionary *)options
    terminationHandler:(void (^)(int status))terminationHandler
                 error:(NSError **)error;
@end

CoreSimulatorSupport::Process::Process(lldb::pid_t p) : m_pid(p), m_error() {}

CoreSimulatorSupport::Process::Process(Error error)
    : m_pid(LLDB_INVALID_PROCESS_ID), m_error(error) {}

CoreSimulatorSupport::Process::Process(lldb::pid_t p, Error error)
    : m_pid(p), m_error(error) {}

CoreSimulatorSupport::DeviceType::DeviceType()
    : m_dev(nil), m_model_identifier() {}

CoreSimulatorSupport::DeviceType::DeviceType(id d)
    : m_dev(d), m_model_identifier() {}

CoreSimulatorSupport::DeviceType::operator bool() { return m_dev != nil; }

ConstString CoreSimulatorSupport::DeviceType::GetIdentifier() {
  return ConstString([[m_dev identifier] UTF8String]);
}

ConstString CoreSimulatorSupport::DeviceType::GetProductFamily() {
  return ConstString([[m_dev productFamily] UTF8String]);
}

CoreSimulatorSupport::DeviceType::ProductFamilyID
CoreSimulatorSupport::DeviceType::GetProductFamilyID() {
  return ProductFamilyID([m_dev productFamilyID]);
}

CoreSimulatorSupport::DeviceRuntime::DeviceRuntime()
    : m_dev(nil), m_os_version() {}

CoreSimulatorSupport::DeviceRuntime::DeviceRuntime(id d)
    : m_dev(d), m_os_version() {}

CoreSimulatorSupport::DeviceRuntime::operator bool() { return m_dev != nil; }

bool CoreSimulatorSupport::DeviceRuntime::IsAvailable() {
  return [m_dev available];
}

CoreSimulatorSupport::Device::Device()
    : m_dev(nil), m_dev_type(), m_dev_runtime() {}

CoreSimulatorSupport::Device::Device(id d)
    : m_dev(d), m_dev_type(), m_dev_runtime() {}

CoreSimulatorSupport::Device::operator bool() { return m_dev != nil; }

CoreSimulatorSupport::Device::State CoreSimulatorSupport::Device::GetState() {
  return (State)([m_dev state]);
}

CoreSimulatorSupport::ModelIdentifier::ModelIdentifier(const std::string &mi)
    : m_family(), m_versions() {
  bool any = false;
  bool first_digit = false;
  unsigned int val = 0;

  for (char c : mi) {
    any = true;
    if (::isdigit(c)) {
      if (!first_digit)
        first_digit = true;
      val = 10 * val + (c - '0');
    } else if (c == ',') {
      if (first_digit) {
        m_versions.push_back(val);
        val = 0;
      } else
        m_family.push_back(c);
    } else {
      if (first_digit) {
        m_family.clear();
        m_versions.clear();
        return;
      } else {
        m_family.push_back(c);
      }
    }
  }

  if (first_digit)
    m_versions.push_back(val);
}

CoreSimulatorSupport::ModelIdentifier::ModelIdentifier()
    : ModelIdentifier("") {}

CoreSimulatorSupport::OSVersion::OSVersion(const std::string &ver,
                                           const std::string &build)
    : m_versions(), m_build(build) {
  bool any = false;
  unsigned int val = 0;
  for (char c : ver) {
    if (c == '.') {
      m_versions.push_back(val);
      val = 0;
    } else if (::isdigit(c)) {
      val = 10 * val + (c - '0');
      any = true;
    } else {
      m_versions.clear();
      return;
    }
  }
  if (any)
    m_versions.push_back(val);
}

CoreSimulatorSupport::OSVersion::OSVersion() : OSVersion("", "") {}

CoreSimulatorSupport::ModelIdentifier
CoreSimulatorSupport::DeviceType::GetModelIdentifier() {
  if (!m_model_identifier.hasValue()) {
    auto utf8_model_id = [[m_dev modelIdentifier] UTF8String];
    if (utf8_model_id && *utf8_model_id)
      m_model_identifier = ModelIdentifier(utf8_model_id);
  }

  if (m_model_identifier.hasValue())
    return m_model_identifier.getValue();
  else
    return ModelIdentifier();
}

CoreSimulatorSupport::OSVersion
CoreSimulatorSupport::DeviceRuntime::GetVersion() {
  if (!m_os_version.hasValue()) {
    auto utf8_ver_string = [[m_dev versionString] UTF8String];
    auto utf8_build_ver = [[m_dev buildVersionString] UTF8String];
    if (utf8_ver_string && *utf8_ver_string && utf8_build_ver &&
        *utf8_build_ver) {
      m_os_version = OSVersion(utf8_ver_string, utf8_build_ver);
    }
  }

  if (m_os_version.hasValue())
    return m_os_version.getValue();
  return OSVersion();
}

std::string CoreSimulatorSupport::DeviceType::GetName() {
  auto utf8_name = [[m_dev name] UTF8String];
  if (utf8_name)
    return std::string(utf8_name);
  return "";
}

std::string CoreSimulatorSupport::Device::GetName() const {
  auto utf8_name = [[m_dev name] UTF8String];
  if (utf8_name)
    return std::string(utf8_name);
  return "";
}

std::string CoreSimulatorSupport::Device::GetUDID() const {
  auto utf8_udid = [[[m_dev UDID] UUIDString] UTF8String];
  if (utf8_udid)
    return std::string(utf8_udid);
  else
    return std::string();
}

CoreSimulatorSupport::DeviceType CoreSimulatorSupport::Device::GetDeviceType() {
  if (!m_dev_type.hasValue())
    m_dev_type = DeviceType([m_dev deviceType]);

  return m_dev_type.getValue();
}

CoreSimulatorSupport::DeviceRuntime
CoreSimulatorSupport::Device::GetDeviceRuntime() {
  if (!m_dev_runtime.hasValue())
    m_dev_runtime = DeviceRuntime([m_dev runtime]);

  return m_dev_runtime.getValue();
}

bool CoreSimulatorSupport::
operator>(const CoreSimulatorSupport::OSVersion &lhs,
          const CoreSimulatorSupport::OSVersion &rhs) {
  for (size_t i = 0; i < rhs.GetNumVersions(); i++) {
    unsigned int l = lhs.GetVersionAtIndex(i);
    unsigned int r = rhs.GetVersionAtIndex(i);
    if (l > r)
      return true;
  }
  return false;
}

bool CoreSimulatorSupport::
operator>(const CoreSimulatorSupport::ModelIdentifier &lhs,
          const CoreSimulatorSupport::ModelIdentifier &rhs) {
  if (lhs.GetFamily() != rhs.GetFamily())
    return false;
  for (size_t i = 0; i < rhs.GetNumVersions(); i++) {
    unsigned int l = lhs.GetVersionAtIndex(i);
    unsigned int r = rhs.GetVersionAtIndex(i);
    if (l > r)
      return true;
  }
  return false;
}

bool CoreSimulatorSupport::
operator<(const CoreSimulatorSupport::OSVersion &lhs,
          const CoreSimulatorSupport::OSVersion &rhs) {
  for (size_t i = 0; i < rhs.GetNumVersions(); i++) {
    unsigned int l = lhs.GetVersionAtIndex(i);
    unsigned int r = rhs.GetVersionAtIndex(i);
    if (l < r)
      return true;
  }
  return false;
}

bool CoreSimulatorSupport::
operator<(const CoreSimulatorSupport::ModelIdentifier &lhs,
          const CoreSimulatorSupport::ModelIdentifier &rhs) {
  if (lhs.GetFamily() != rhs.GetFamily())
    return false;

  for (size_t i = 0; i < rhs.GetNumVersions(); i++) {
    unsigned int l = lhs.GetVersionAtIndex(i);
    unsigned int r = rhs.GetVersionAtIndex(i);
    if (l < r)
      return true;
  }
  return false;
}

bool CoreSimulatorSupport::
operator==(const CoreSimulatorSupport::OSVersion &lhs,
           const CoreSimulatorSupport::OSVersion &rhs) {
  for (size_t i = 0; i < rhs.GetNumVersions(); i++) {
    unsigned int l = lhs.GetVersionAtIndex(i);
    unsigned int r = rhs.GetVersionAtIndex(i);
    if (l != r)
      return false;
  }
  return true;
}

bool CoreSimulatorSupport::
operator==(const CoreSimulatorSupport::ModelIdentifier &lhs,
           const CoreSimulatorSupport::ModelIdentifier &rhs) {
  if (lhs.GetFamily() != rhs.GetFamily())
    return false;

  for (size_t i = 0; i < rhs.GetNumVersions(); i++) {
    unsigned int l = lhs.GetVersionAtIndex(i);
    unsigned int r = rhs.GetVersionAtIndex(i);
    if (l != r)
      return false;
  }
  return true;
}

bool CoreSimulatorSupport::
operator!=(const CoreSimulatorSupport::OSVersion &lhs,
           const CoreSimulatorSupport::OSVersion &rhs) {
  for (size_t i = 0; i < rhs.GetNumVersions(); i++) {
    unsigned int l = lhs.GetVersionAtIndex(i);
    unsigned int r = rhs.GetVersionAtIndex(i);
    if (l != r)
      return true;
  }
  return false;
}

bool CoreSimulatorSupport::
operator!=(const CoreSimulatorSupport::ModelIdentifier &lhs,
           const CoreSimulatorSupport::ModelIdentifier &rhs) {
  if (lhs.GetFamily() != rhs.GetFamily())
    return false;

  for (size_t i = 0; i < rhs.GetNumVersions(); i++) {
    unsigned int l = lhs.GetVersionAtIndex(i);
    unsigned int r = rhs.GetVersionAtIndex(i);
    if (l != r)
      return true;
  }
  return false;
}

bool CoreSimulatorSupport::Device::Boot(Error &err) {
  if (m_dev == nil) {
    err.SetErrorString("no valid simulator instance");
    return false;
  }

#define kSimDeviceBootPersist                                                  \
  @"persist" /* An NSNumber (boolean) indicating whether or not the session    \
                should outlive the calling process (default false) */

  NSDictionary *options = @{
    kSimDeviceBootPersist : @NO,
  };

#undef kSimDeviceBootPersist

  NSError *nserror;
  if ([m_dev bootWithOptions:options error:&nserror]) {
    err.Clear();
    return true;
  } else {
    err.SetErrorString([[nserror description] UTF8String]);
    return false;
  }
}

bool CoreSimulatorSupport::Device::Shutdown(Error &err) {
  NSError *nserror;
  if ([m_dev shutdownWithError:&nserror]) {
    err.Clear();
    return true;
  } else {
    err.SetErrorString([[nserror description] UTF8String]);
    return false;
  }
}

static Error HandleFileAction(ProcessLaunchInfo &launch_info,
                              NSMutableDictionary *options, NSString *key,
                              const int fd, File &file) {
  Error error;
  const FileAction *file_action = launch_info.GetFileActionForFD(fd);
  if (file_action) {
    switch (file_action->GetAction()) {
    case FileAction::eFileActionNone:
      break;

    case FileAction::eFileActionClose:
      error.SetErrorStringWithFormat("close file action for %i not supported",
                                     fd);
      break;

    case FileAction::eFileActionDuplicate:
      error.SetErrorStringWithFormat(
          "duplication file action for %i not supported", fd);
      break;

    case FileAction::eFileActionOpen: {
      FileSpec file_spec = file_action->GetFileSpec();
      if (file_spec) {
        const int master_fd = launch_info.GetPTY().GetMasterFileDescriptor();
        if (master_fd != PseudoTerminal::invalid_fd) {
          // Check in case our file action open wants to open the slave
          const char *slave_path = launch_info.GetPTY().GetSlaveName(NULL, 0);
          if (slave_path) {
            FileSpec slave_spec(slave_path, false);
            if (file_spec == slave_spec) {
              int slave_fd = launch_info.GetPTY().GetSlaveFileDescriptor();
              if (slave_fd == PseudoTerminal::invalid_fd)
                slave_fd = launch_info.GetPTY().OpenSlave(O_RDWR, nullptr, 0);
              if (slave_fd == PseudoTerminal::invalid_fd) {
                error.SetErrorStringWithFormat("unable to open slave pty '%s'",
                                               slave_path);
                return error; // Failure
              }
              [options setValue:[NSNumber numberWithInteger:slave_fd]
                         forKey:key];
              return error; // Success
            }
          }
        }
        Error posix_error;
        int created_fd =
            open(file_spec.GetPath().c_str(), file_action->GetActionArgument(),
                 S_IRUSR | S_IWUSR);
        if (created_fd >= 0) {
          file.SetDescriptor(created_fd, true);
          [options setValue:[NSNumber numberWithInteger:created_fd] forKey:key];
          return error; // Success
        } else {
          posix_error.SetErrorToErrno();
          error.SetErrorStringWithFormat("unable to open file '%s': %s",
                                         file_spec.GetPath().c_str(),
                                         posix_error.AsCString());
        }
      }
    } break;
    }
  }
  return error; // Success, no file action, nothing to do
}

CoreSimulatorSupport::Process
CoreSimulatorSupport::Device::Spawn(ProcessLaunchInfo &launch_info) {
#define kSimDeviceSpawnEnvironment                                             \
  @"environment" /* An NSDictionary (NSStrings -> NSStrings) of environment    \
                    key/values */
#define kSimDeviceSpawnStdin @"stdin"   /* An NSNumber corresponding to a fd */
#define kSimDeviceSpawnStdout @"stdout" /* An NSNumber corresponding to a fd   \
                                           */
#define kSimDeviceSpawnStderr @"stderr" /* An NSNumber corresponding to a fd   \
                                           */
#define kSimDeviceSpawnArguments                                               \
  @"arguments" /* An NSArray of strings to use as the argv array.  If not      \
                  provided, path will be argv[0] */
#define kSimDeviceSpawnWaitForDebugger                                         \
  @"wait_for_debugger" /* An NSNumber (bool) */

  NSMutableDictionary *options = [[NSMutableDictionary alloc] init];

  if (launch_info.GetFlags().Test(lldb::eLaunchFlagDebug))
    [options setObject:@YES forKey:kSimDeviceSpawnWaitForDebugger];

  if (launch_info.GetArguments().GetArgumentCount()) {
    const Args &args(launch_info.GetArguments());
    NSMutableArray *args_array = [[NSMutableArray alloc] init];
    for (size_t idx = 0; idx < args.GetArgumentCount(); idx++)
      [args_array
          addObject:[NSString
                        stringWithUTF8String:args.GetArgumentAtIndex(idx)]];

    [options setObject:args_array forKey:kSimDeviceSpawnArguments];
  }

  if (launch_info.GetEnvironmentEntries().GetArgumentCount()) {
    const Args &envs(launch_info.GetEnvironmentEntries());
    NSMutableDictionary *env_dict = [[NSMutableDictionary alloc] init];
    for (size_t idx = 0; idx < envs.GetArgumentCount(); idx++) {
      llvm::StringRef arg_sr(envs.GetArgumentAtIndex(idx));
      auto first_eq = arg_sr.find('=');
      if (first_eq == llvm::StringRef::npos)
        continue;
      llvm::StringRef key = arg_sr.substr(0, first_eq);
      llvm::StringRef value = arg_sr.substr(first_eq + 1);

      NSString *key_ns = [NSString stringWithUTF8String:key.str().c_str()];
      NSString *value_ns = [NSString stringWithUTF8String:value.str().c_str()];

      [env_dict setValue:value_ns forKey:key_ns];
    }

    [options setObject:env_dict forKey:kSimDeviceSpawnEnvironment];
  }

  Error error;
  File stdin_file;
  File stdout_file;
  File stderr_file;
  error = HandleFileAction(launch_info, options, kSimDeviceSpawnStdin,
                           STDIN_FILENO, stdin_file);

  if (error.Fail())
    return CoreSimulatorSupport::Process(error);

  error = HandleFileAction(launch_info, options, kSimDeviceSpawnStdout,
                           STDOUT_FILENO, stdout_file);

  if (error.Fail())
    return CoreSimulatorSupport::Process(error);

  error = HandleFileAction(launch_info, options, kSimDeviceSpawnStderr,
                           STDERR_FILENO, stderr_file);

  if (error.Fail())
    return CoreSimulatorSupport::Process(error);

#undef kSimDeviceSpawnEnvironment
#undef kSimDeviceSpawnStdin
#undef kSimDeviceSpawnStdout
#undef kSimDeviceSpawnStderr
#undef kSimDeviceSpawnWaitForDebugger
#undef kSimDeviceSpawnArguments

  NSError *nserror;

  pid_t pid = [m_dev
           spawnWithPath:[NSString stringWithUTF8String:launch_info
                                                            .GetExecutableFile()
                                                            .GetPath()
                                                            .c_str()]
                 options:options
      terminationHandler:nil
                   error:&nserror];

  if (pid < 0) {
    const char *nserror_string = [[nserror description] UTF8String];
    error.SetErrorString(nserror_string ? nserror_string : "unable to launch");
  }

  return CoreSimulatorSupport::Process(pid, error);
}

CoreSimulatorSupport::DeviceSet
CoreSimulatorSupport::DeviceSet::GetAllDevices(const char *developer_dir) {
  if (!developer_dir || !developer_dir[0])
    return DeviceSet([NSArray new]);

  Class SimServiceContextClass = NSClassFromString(@"SimServiceContext");
  NSString *dev_dir = @(developer_dir);
  NSError *error = nil;

  id serviceContext =
      [SimServiceContextClass sharedServiceContextForDeveloperDir:dev_dir
                                                            error:&error];
  if (!serviceContext)
    return DeviceSet([NSArray new]);

  return DeviceSet([[serviceContext defaultDeviceSetWithError:&error] devices]);
}

CoreSimulatorSupport::DeviceSet
CoreSimulatorSupport::DeviceSet::GetAvailableDevices(
    const char *developer_dir) {
  return GetAllDevices(developer_dir).GetDevicesIf([](Device d) -> bool {
    return (d && d.GetDeviceType() && d.GetDeviceRuntime() &&
            d.GetDeviceRuntime().IsAvailable());
  });
}

size_t CoreSimulatorSupport::DeviceSet::GetNumDevices() {
  return [m_dev count];
}

CoreSimulatorSupport::Device
CoreSimulatorSupport::DeviceSet::GetDeviceAtIndex(size_t idx) {
  if (idx < GetNumDevices())
    return Device([m_dev objectAtIndex:idx]);
  return Device();
}

CoreSimulatorSupport::DeviceSet CoreSimulatorSupport::DeviceSet::GetDevicesIf(
    std::function<bool(CoreSimulatorSupport::Device)> f) {
  NSMutableArray *array = [[NSMutableArray alloc] init];
  for (NSUInteger i = 0; i < GetNumDevices(); i++) {
    Device d(GetDeviceAtIndex(i));
    if (f(d))
      [array addObject:(id)d.m_dev];
  }

  return DeviceSet(array);
}

void CoreSimulatorSupport::DeviceSet::ForEach(
    std::function<bool(const Device &)> f) {
  const size_t n = GetNumDevices();
  for (NSUInteger i = 0; i < n; ++i) {
    if (f(GetDeviceAtIndex(i)) == false)
      break;
  }
}

CoreSimulatorSupport::DeviceSet CoreSimulatorSupport::DeviceSet::GetDevices(
    CoreSimulatorSupport::DeviceType::ProductFamilyID dev_id) {
  NSMutableArray *array = [[NSMutableArray alloc] init];
  const size_t n = GetNumDevices();
  for (NSUInteger i = 0; i < n; ++i) {
    Device d(GetDeviceAtIndex(i));
    if (d && d.GetDeviceType() &&
        d.GetDeviceType().GetProductFamilyID() == dev_id)
      [array addObject:(id)d.m_dev];
  }

  return DeviceSet(array);
}

CoreSimulatorSupport::Device CoreSimulatorSupport::DeviceSet::GetFanciest(
    CoreSimulatorSupport::DeviceType::ProductFamilyID dev_id) {
  Device dev;

  for (NSUInteger i = 0; i < GetNumDevices(); i++) {
    Device d(GetDeviceAtIndex(i));
    if (d && d.GetDeviceType() &&
        d.GetDeviceType().GetProductFamilyID() == dev_id) {
      if (!dev)
        dev = d;
      else {
        if ((d.GetDeviceType().GetModelIdentifier() >
             dev.GetDeviceType().GetModelIdentifier()) ||
            d.GetDeviceRuntime().GetVersion() >
                dev.GetDeviceRuntime().GetVersion())
          dev = d;
      }
    }
  }

  return dev;
}
