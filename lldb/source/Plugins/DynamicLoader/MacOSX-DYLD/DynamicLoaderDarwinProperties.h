//===-- DynamicLoaderDarwinProperties.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_DYNAMICLOADER_MACOSX_DYLD_DYNAMICLOADERDARWINPROPERTIES_H
#define LLDB_SOURCE_PLUGINS_DYNAMICLOADER_MACOSX_DYLD_DYNAMICLOADERDARWINPROPERTIES_H

#include "lldb/Core/UserSettingsController.h"

namespace lldb_private {

class DynamicLoaderDarwinProperties : public Properties {
public:
  class ExperimentalProperties : public Properties {
  public:
    ExperimentalProperties();
  };
  static llvm::StringRef GetSettingName();
  static DynamicLoaderDarwinProperties &GetGlobal();
  DynamicLoaderDarwinProperties();
  ~DynamicLoaderDarwinProperties() override = default;
  bool GetEnableParallelImageLoad() const;

private:
  std::unique_ptr<ExperimentalProperties> m_experimental_properties;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_DYNAMICLOADER_MACOSX_DYLD_DYNAMICLOADERDARWINPROPERTIES_H
