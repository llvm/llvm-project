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


std::tuple<XcodeSDK::Type, llvm::VersionTuple> XcodeSDK::Parse() const {
  llvm::StringRef input(m_name);
  XcodeSDK::Type sdk = ParseSDKName(input);
  llvm::VersionTuple version = ParseSDKVersion(input);
  return {sdk, version};
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

void XcodeSDK::Merge(XcodeSDK other) {
  // The "bigger" SDK always wins.
  if (Parse() < other.Parse())
    *this = other;
}

llvm::StringRef XcodeSDK::GetSDKNameForType(XcodeSDK::Type type) {
  switch (type) {
  case MacOSX:
    return "macosx";
  case iPhoneSimulator:
    return "iphonesimulator";
  case iPhoneOS:
    return "iphoneos";
  case AppleTVSimulator:
    return "appletvsimulator";
  case AppleTVOS:
    return "appletvos";
  case WatchSimulator:
    return "watchsimulator";
  case watchOS:
    return "watchos";
  case bridgeOS:
    return "bridgeos";
  case Linux:
    return "linux";
  case numSDKTypes:
  case unknown:
    return "";
  }
  llvm_unreachable("unhandled switch case");
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
    const llvm::StringRef sdk_string = GetSDKNameForType(desired_type);
    if (!llvm::StringRef(sdk_name_lower).startswith(sdk_string))
      return false;

    auto version_part = sdk_name.drop_front(sdk_string.size());
    version_part.consume_back(".sdk");

    llvm::VersionTuple version;
    if (version.tryParse(version_part))
      return false;
    return SDKSupportsModules(desired_type, version);
  }

  return false;
}
