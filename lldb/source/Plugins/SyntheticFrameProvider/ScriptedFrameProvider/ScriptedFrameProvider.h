//===-- ScriptedFrameProvider.h --------------------------------*- C++ -*-===//
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
  CreateInstance(lldb::ThreadSP thread_sp);

  static void Initialize();

  static void Terminate();

  /// Constructor that initializes the scripted frame provider.
  ///
  /// \param[in] thread_sp
  ///     The thread for which to provide scripted frames.
  ///
  /// \param[in] scripted_metadata
  ///     The metadata containing the class name and arguments for the
  ///     scripted frame provider.
  ///
  /// \param[out] error
  ///     Status object to report any errors during initialization.
  ScriptedFrameProvider(lldb::ThreadSP thread_sp,
                        const ScriptedMetadata &scripted_metadata,
                        Status &error);
  ~ScriptedFrameProvider() override;

  // PluginInterface methods
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  /// Get a single stack frame at the specified index.
  llvm::Expected<lldb::StackFrameSP>
  GetFrameAtIndex(lldb::StackFrameListSP real_frames, uint32_t idx) override;

private:
  lldb::ScriptedFrameProviderInterfaceSP m_interface_sp;
};

} // namespace lldb_private

#endif // LLDB_PLUGINS_SYNTHETICFRAMEPROVIDER_SCRIPTEDFRAMEPROVIDER_SCRIPTEDFRAMEPROVIDER_H
