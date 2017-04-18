//===-- PlatformRemoteiOS.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformRemoteiOS_h_
#define liblldb_PlatformRemoteiOS_h_

// C Includes
// C++ Includes
#include <string>

// Other libraries and framework includes
// Project includes
#include "PlatformRemoteDarwinDevice.h"
#include "lldb/Utility/FileSpec.h"

#include "llvm/Support/FileSystem.h"

class PlatformRemoteiOS : public PlatformRemoteDarwinDevice {
public:
  PlatformRemoteiOS();

  ~PlatformRemoteiOS() override = default;

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

  bool GetSupportedArchitectureAtIndex(uint32_t idx,
                                       lldb_private::ArchSpec &arch) override;

protected:

  //------------------------------------------------------------
  // lldb_private::PlatformRemoteDarwinDevice functions
  //------------------------------------------------------------

  void GetDeviceSupportDirectoryNames (std::vector<std::string> &dirnames) override;

  std::string GetPlatformName () override;

private:
  DISALLOW_COPY_AND_ASSIGN(PlatformRemoteiOS);
};

#endif // liblldb_PlatformRemoteiOS_h_
