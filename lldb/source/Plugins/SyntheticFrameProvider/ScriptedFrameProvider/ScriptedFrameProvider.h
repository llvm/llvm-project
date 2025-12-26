//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_SYNTHETICFRAMEPROVIDER_SCRIPTEDFRAMEPROVIDER_SCRIPTEDFRAMEPROVIDER_H
#define LLDB_PLUGINS_SYNTHETICFRAMEPROVIDER_SCRIPTEDFRAMEPROVIDER_SCRIPTEDFRAMEPROVIDER_H

#include "lldb/Target/SyntheticFrameProvider.h"
#include "lldb/Utility/ScriptedMetadata.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-forward.h"
#include "llvm/Support/Error.h"

namespace lldb_private {

class ScriptedFrameProvider : public SyntheticFrameProvider {
public:
  static llvm::StringRef GetPluginNameStatic() {
    return "ScriptedFrameProvider";
  }

  static llvm::Expected<lldb::SyntheticFrameProviderSP>
  CreateInstance(lldb::StackFrameListSP input_frames,
                 const ScriptedFrameProviderDescriptor &descriptor);

  static void Initialize();

  static void Terminate();

  ScriptedFrameProvider(lldb::StackFrameListSP input_frames,
                        lldb::ScriptedFrameProviderInterfaceSP interface_sp,
                        const ScriptedFrameProviderDescriptor &descriptor);
  ~ScriptedFrameProvider() override;

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  std::string GetDescription() const override;

  /// Get a single stack frame at the specified index.
  llvm::Expected<lldb::StackFrameSP> GetFrameAtIndex(uint32_t idx) override;

private:
  lldb::ScriptedFrameProviderInterfaceSP m_interface_sp;
  const ScriptedFrameProviderDescriptor &m_descriptor;
};

} // namespace lldb_private

#endif // LLDB_PLUGINS_SYNTHETICFRAMEPROVIDER_SCRIPTEDFRAMEPROVIDER_SCRIPTEDFRAMEPROVIDER_H
