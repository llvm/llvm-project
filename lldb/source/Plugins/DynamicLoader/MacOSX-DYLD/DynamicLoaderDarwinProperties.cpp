//===-- DynamicLoaderDarwinProperties.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DynamicLoaderDarwinProperties.h"

using namespace lldb_private;

#define LLDB_PROPERTIES_dynamicloaderdarwin_experimental
#include "DynamicLoaderDarwinProperties.inc"

enum {
#define LLDB_PROPERTIES_dynamicloaderdarwin_experimental
#include "DynamicLoaderDarwinPropertiesEnum.inc"
};

llvm::StringRef DynamicLoaderDarwinProperties::GetSettingName() {
  static constexpr llvm::StringLiteral g_setting_name("darwin");
  return g_setting_name;
}

DynamicLoaderDarwinProperties::ExperimentalProperties::ExperimentalProperties()
    : Properties(std::make_shared<OptionValueProperties>(
          GetExperimentalSettingsName())) {
  m_collection_sp->Initialize(g_dynamicloaderdarwin_experimental_properties);
}

DynamicLoaderDarwinProperties::DynamicLoaderDarwinProperties()
    : Properties(std::make_shared<OptionValueProperties>(GetSettingName())),
      m_experimental_properties(std::make_unique<ExperimentalProperties>()) {
  m_collection_sp->AppendProperty(
      Properties::GetExperimentalSettingsName(),
      "Experimental settings - setting these won't produce errors if the "
      "setting is not present.",
      true, m_experimental_properties->GetValueProperties());
}

bool DynamicLoaderDarwinProperties::GetEnableParallelImageLoad() const {
  return m_experimental_properties->GetPropertyAtIndexAs<bool>(
      ePropertyEnableParallelImageLoad,
      g_dynamicloaderdarwin_experimental_properties
              [ePropertyEnableParallelImageLoad]
                  .default_uint_value != 0);
}

DynamicLoaderDarwinProperties &DynamicLoaderDarwinProperties::GetGlobal() {
  static DynamicLoaderDarwinProperties g_settings;
  return g_settings;
}
