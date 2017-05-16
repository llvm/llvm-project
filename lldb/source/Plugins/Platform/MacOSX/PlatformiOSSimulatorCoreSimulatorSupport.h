//===-- PlatformiOSSimulatorCoreSimulatorSupport.h ----------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformiOSSimulatorCoreSimulatorSupport_h_
#define liblldb_PlatformiOSSimulatorCoreSimulatorSupport_h_

// C Includes
// C++ Includes
#include <functional>
#include <ostream>
#include <string>
#include <vector>
// Other libraries and framework includes
#ifdef __APPLE__
#include <objc/objc.h>
#else
typedef void *id;
#endif
// Project includes
#include "lldb/Interpreter/Args.h"
#include "lldb/Target/ProcessLaunchInfo.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Status.h"

#include "llvm/ADT/Optional.h"

// And now the actual magic
namespace CoreSimulatorSupport {
class Process {
public:
  lldb::pid_t GetPID() { return m_pid; }

  explicit operator bool() { return m_pid != LLDB_INVALID_PROCESS_ID; }

  lldb_private::Status GetError() { return m_error; }

private:
  Process(lldb::pid_t p);

  Process(lldb_private::Status error);

  Process(lldb::pid_t p, lldb_private::Status error);

  lldb::pid_t m_pid;
  lldb_private::Status m_error;

  friend class Device;
};

class ModelIdentifier {
public:
  ModelIdentifier(const std::string &mi);
  ModelIdentifier();

  explicit operator bool() const { return !m_versions.empty(); }

  size_t GetNumVersions() const { return m_versions.size(); }

  unsigned int GetVersionAtIndex(size_t idx) const { return m_versions[idx]; }

  std::string GetFamily() const { return m_family.c_str(); }

private:
  std::string m_family;
  std::vector<unsigned int> m_versions;
};

class DeviceType {
public:
  enum class ProductFamilyID : int32_t {
    iPhone = 1,
    iPad = 2,
    appleTV = 3,
    appleWatch = 4
  };

  DeviceType();

  DeviceType(id d);

  explicit operator bool();

  std::string GetName();

  lldb_private::ConstString GetIdentifier();

  ModelIdentifier GetModelIdentifier();

  lldb_private::ConstString GetProductFamily();

  ProductFamilyID GetProductFamilyID();

private:
  id m_dev;
  llvm::Optional<ModelIdentifier> m_model_identifier;
};

class OSVersion {
public:
  OSVersion(const std::string &ver, const std::string &build);

  OSVersion();

  explicit operator bool() const { return !m_versions.empty(); }

  size_t GetNumVersions() const { return m_versions.size(); }

  unsigned int GetVersionAtIndex(size_t idx) const { return m_versions[idx]; }

  const char *GetBuild() const { return m_build.c_str(); }

private:
  std::vector<unsigned int> m_versions;
  std::string m_build;
};

class DeviceRuntime {
public:
  DeviceRuntime();

  DeviceRuntime(id d);

  explicit operator bool();

  OSVersion GetVersion();

  bool IsAvailable();

private:
  id m_dev;
  llvm::Optional<OSVersion> m_os_version;
};

class Device {
private:
  typedef unsigned long int NSUInteger;

public:
  enum class State : NSUInteger {
    Creating,
    Shutdown,
    Booting,
    Booted,
    ShuttingDown
  };

  Device();

  Device(id d);

  explicit operator bool();

  std::string GetName() const;

  DeviceType GetDeviceType();

  DeviceRuntime GetDeviceRuntime();

  State GetState();

  bool Boot(lldb_private::Status &err);

  bool Shutdown(lldb_private::Status &err);

  std::string GetUDID() const;

  Process Spawn(lldb_private::ProcessLaunchInfo &launch_info);

private:
  id m_dev;
  llvm::Optional<DeviceType> m_dev_type;
  llvm::Optional<DeviceRuntime> m_dev_runtime;

  friend class DeviceSet;
};

bool operator>(const OSVersion &lhs, const OSVersion &rhs);

bool operator>(const ModelIdentifier &lhs, const ModelIdentifier &rhs);

bool operator<(const OSVersion &lhs, const OSVersion &rhs);

bool operator<(const ModelIdentifier &lhs, const ModelIdentifier &rhs);

bool operator==(const OSVersion &lhs, const OSVersion &rhs);

bool operator==(const ModelIdentifier &lhs, const ModelIdentifier &rhs);

bool operator!=(const OSVersion &lhs, const OSVersion &rhs);

bool operator!=(const ModelIdentifier &lhs, const ModelIdentifier &rhs);

class DeviceSet {
public:
  static DeviceSet GetAllDevices(const char *developer_dir);

  static DeviceSet GetAvailableDevices(const char *developer_dir);

  size_t GetNumDevices();

  Device GetDeviceAtIndex(size_t idx);

  void ForEach(std::function<bool(const Device &)> f);

  DeviceSet GetDevicesIf(std::function<bool(Device)> f);

  DeviceSet GetDevices(DeviceType::ProductFamilyID dev_id);

  Device GetFanciest(DeviceType::ProductFamilyID dev_id);

private:
  DeviceSet(id arr) : m_dev(arr) {}

  id m_dev;
};
}

#endif // liblldb_PlatformiOSSimulatorCoreSimulatorSupport_h_
