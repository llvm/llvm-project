//===-- XcodeSDK.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/XcodeSDK.h"

#include "lldb/lldb-types.h"
#include <string>

using namespace lldb;
using namespace lldb_private;

XcodeSDK &XcodeSDK::operator=(XcodeSDK other) {
  m_name = other.m_name;
  return *this;
}

bool XcodeSDK::operator==(XcodeSDK other) {
  return m_name == other.m_name;
}

static XcodeSDK::Type ParseSDKName(llvm::StringRef &name) {
  if (name.consume_front("MacOSX"))
    return XcodeSDK::MacOSX;
  if (name.consume_front("iPhoneSimulator"))
    return XcodeSDK::iPhoneSimulator;
  if (name.consume_front("iPhoneOS"))
    return XcodeSDK::iPhoneOS;
  if (name.consume_front("AppleTVSimulator"))
    return XcodeSDK::AppleTVSimulator;
  if (name.consume_front("AppleTVOS"))
    return XcodeSDK::AppleTVOS;
  if (name.consume_front("WatchSimulator"))
    return XcodeSDK::WatchSimulator;
  if (name.consume_front("WatchOS"))
    return XcodeSDK::watchOS;
  if (name.consume_front("bridgeOS"))
    return XcodeSDK::bridgeOS;
  if (name.consume_front("Linux"))
    return XcodeSDK::Linux;
  static_assert(XcodeSDK::Linux == XcodeSDK::numSDKTypes - 1,
                "New SDK type was added, update this list!");
  return XcodeSDK::unknown;
}

static llvm::VersionTuple ParseSDKVersion(llvm::StringRef &name) {
  unsigned i = 0;
  while (i < name.size() && name[i] >= '0' && name[i] <= '9')
    ++i;
  if (i == name.size() || name[i++] != '.')
    return {};
  while (i < name.size() && name[i] >= '0' && name[i] <= '9')
    ++i;
  if (i == name.size() || name[i++] != '.')
    return {};

  llvm::VersionTuple version;
  version.tryParse(name.slice(0, i - 1));
  name = name.drop_front(i);
  return version;
}

static bool ParseAppleInternalSDK(llvm::StringRef &name) {
  return name.consume_front("Internal.");
}

XcodeSDK::Info XcodeSDK::Parse() const {
  XcodeSDK::Info info;
  llvm::StringRef input(m_name);
  info.type = ParseSDKName(input);
  info.version = ParseSDKVersion(input);
  info.internal = ParseAppleInternalSDK(input);
  return info;
}

bool XcodeSDK::IsAppleInternalSDK() const {
  llvm::StringRef input(m_name);
  ParseSDKName(input);
  ParseSDKVersion(input);
  return ParseAppleInternalSDK(input);
}

llvm::VersionTuple XcodeSDK::GetVersion() const {
  llvm::StringRef input(m_name);
  ParseSDKName(input);
  return ParseSDKVersion(input);
}

XcodeSDK::Type XcodeSDK::GetType() const {
  llvm::StringRef input(m_name);
  return ParseSDKName(input);
}

llvm::StringRef XcodeSDK::GetString() const { return m_name; }

bool XcodeSDK::Info::operator<(const Info &other) const {
  return std::tie(type, version, internal) <
         std::tie(other.type, other.version, other.internal);
}
void XcodeSDK::Merge(XcodeSDK other) {
  // The "bigger" SDK always wins.
  auto l = Parse();
  auto r = other.Parse();
  if (l < r)
    *this = other;
  else {
    // The Internal flag always wins.
    if (llvm::StringRef(m_name).endswith(".sdk"))
      if (!l.internal && r.internal)
        m_name =
            m_name.substr(0, m_name.size() - 3) + std::string("Internal.sdk");
  }
}

std::string XcodeSDK::GetCanonicalName(XcodeSDK::Info info) {
  std::string name;
  switch (info.type) {
  case MacOSX:
    name = "macosx";
    break;
  case iPhoneSimulator:
    name = "iphonesimulator";
    break;
  case iPhoneOS:
    name = "iphoneos";
    break;
  case AppleTVSimulator:
    name = "appletvsimulator";
    break;
  case AppleTVOS:
    name = "appletvos";
    break;
  case WatchSimulator:
    name = "watchsimulator";
    break;
  case watchOS:
    name = "watchos";
    break;
  case bridgeOS:
    name = "bridgeos";
    break;
  case Linux:
    name = "linux";
    break;
  case numSDKTypes:
  case unknown:
    return {};
  }
  if (!info.version.empty())
    name += info.version.getAsString();
  if (info.internal)
    name += ".internal";
  return name;
}

bool XcodeSDK::SDKSupportsModules(XcodeSDK::Type sdk_type,
                                  llvm::VersionTuple version) {
  switch (sdk_type) {
  case Type::MacOSX:
    return version >= llvm::VersionTuple(10, 10);
  case Type::iPhoneOS:
  case Type::iPhoneSimulator:
  case Type::AppleTVOS:
  case Type::AppleTVSimulator:
    return version >= llvm::VersionTuple(8);
  case Type::watchOS:
  case Type::WatchSimulator:
    return version >= llvm::VersionTuple(6);
  default:
    return false;
  }

  return false;
}

bool XcodeSDK::SDKSupportsModules(XcodeSDK::Type desired_type,
                                  const FileSpec &sdk_path) {
  ConstString last_path_component = sdk_path.GetLastPathComponent();

  if (last_path_component) {
    const llvm::StringRef sdk_name = last_path_component.GetStringRef();

    const std::string sdk_name_lower = sdk_name.lower();
    Info info;
    info.type = desired_type;
    const std::string sdk_string = GetCanonicalName(info);
    if (!llvm::StringRef(sdk_name_lower).startswith(sdk_string))
      return false;

    auto version_part = sdk_name.drop_front(sdk_string.size());
    version_part.consume_back(".sdk");
    version_part.consume_back(".Internal");

    llvm::VersionTuple version;
    if (version.tryParse(version_part))
      return false;
    return SDKSupportsModules(desired_type, version);
  }

  return false;
}
