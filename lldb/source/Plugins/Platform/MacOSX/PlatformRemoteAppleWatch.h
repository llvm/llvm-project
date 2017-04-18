//===-- PlatformRemoteAppleWatch.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformRemoteAppleWatch_h_
#define liblldb_PlatformRemoteAppleWatch_h_

// C Includes
// C++ Includes
#include <string>
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/Utility/FileSpec.h"

#include "PlatformRemoteDarwinDevice.h"

#include "llvm/Support/FileSystem.h"

class PlatformRemoteAppleWatch : public PlatformRemoteDarwinDevice {
public:
  PlatformRemoteAppleWatch();

  ~PlatformRemoteAppleWatch() override = default;

  //------------------------------------------------------------
  // Class Functions
  //------------------------------------------------------------
  static lldb::PlatformSP CreateInstance(bool force,
                                         const lldb_private::ArchSpec *arch);

  static void Initialize();

  static void Terminate();

  static lldb_private::ConstString GetPluginNameStatic();

  static const char *GetDescriptionStatic();

  //------------------------------------------------------------
  // lldb_private::Platform functions
  //------------------------------------------------------------

  const char *GetDescription() override { return GetDescriptionStatic(); }

  //------------------------------------------------------------
  // lldb_private::PluginInterface functions
  //------------------------------------------------------------
  lldb_private::ConstString GetPluginName() override {
    return GetPluginNameStatic();
  }

  uint32_t GetPluginVersion() override { return 1; }

  //------------------------------------------------------------
  // lldb_private::Platform functions
  //------------------------------------------------------------

  bool GetSupportedArchitectureAtIndex(uint32_t idx,
                                       lldb_private::ArchSpec &arch) override;

protected:

  //------------------------------------------------------------
  // lldb_private::PlatformRemoteDarwinDevice functions
  //------------------------------------------------------------

  void GetDeviceSupportDirectoryNames (std::vector<std::string> &dirnames) override;

  std::string GetPlatformName () override;

private:
  DISALLOW_COPY_AND_ASSIGN(PlatformRemoteAppleWatch);
};

#endif // liblldb_PlatformRemoteAppleWatch_h_
