//===-- XcodeSDKPath.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/XcodeSDKPath.h"

#include <string>

using namespace lldb;
using namespace lldb_private;

void XcodeSDKPath::Merge(const XcodeSDKPath &other) {
  m_sdk.Merge(other.m_sdk);

  // We changed the SDK name. Adjust the sysroot accordingly.
  auto name = m_sdk.GetString();
  if (m_sysroot && m_sysroot.GetFilename().GetStringRef() != name)
    m_sysroot.SetFilename(name);
}

XcodeSDK XcodeSDKPath::TakeSDK() const { return std::move(m_sdk); }

bool XcodeSDKPath::operator==(const XcodeSDKPath &other) const {
  return m_sdk == other.m_sdk && m_sysroot == other.m_sysroot;
}
