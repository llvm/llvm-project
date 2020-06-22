//===-- XcodeSDKTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/XcodeSDK.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"

#include <tuple>

using namespace lldb_private;

TEST(XcodeSDKTest, ParseTest) {
  EXPECT_EQ(XcodeSDK::GetAnyMacOS().GetType(), XcodeSDK::MacOSX);
  EXPECT_EQ(XcodeSDK("MacOSX.sdk").GetType(), XcodeSDK::MacOSX);
  EXPECT_EQ(XcodeSDK("iPhoneSimulator.sdk").GetType(), XcodeSDK::iPhoneSimulator);
  EXPECT_EQ(XcodeSDK("iPhoneOS.sdk").GetType(), XcodeSDK::iPhoneOS);
  EXPECT_EQ(XcodeSDK("AppleTVSimulator.sdk").GetType(), XcodeSDK::AppleTVSimulator);
  EXPECT_EQ(XcodeSDK("AppleTVOS.sdk").GetType(), XcodeSDK::AppleTVOS);
  EXPECT_EQ(XcodeSDK("WatchSimulator.sdk").GetType(), XcodeSDK::WatchSimulator);
  EXPECT_EQ(XcodeSDK("WatchOS.sdk").GetType(), XcodeSDK::watchOS);
  EXPECT_EQ(XcodeSDK("Linux.sdk").GetType(), XcodeSDK::Linux);
  EXPECT_EQ(XcodeSDK("MacOSX.sdk").GetVersion(), llvm::VersionTuple());
  EXPECT_EQ(XcodeSDK("MacOSX10.9.sdk").GetVersion(), llvm::VersionTuple(10, 9));
  EXPECT_EQ(XcodeSDK("MacOSX10.15.4.sdk").GetVersion(), llvm::VersionTuple(10, 15));
  EXPECT_EQ(XcodeSDK("MacOSX.sdk").IsAppleInternalSDK(), false);
  EXPECT_EQ(XcodeSDK("MacOSX10.15.Internal.sdk").GetType(), XcodeSDK::MacOSX);
  EXPECT_EQ(XcodeSDK("MacOSX10.15.Internal.sdk").GetVersion(),
            llvm::VersionTuple(10, 15));
  EXPECT_EQ(XcodeSDK("MacOSX10.15.Internal.sdk").IsAppleInternalSDK(), true);
  EXPECT_EQ(XcodeSDK().GetType(), XcodeSDK::unknown);
  EXPECT_EQ(XcodeSDK().GetVersion(), llvm::VersionTuple());
}

TEST(XcodeSDKTest, MergeTest) {
  XcodeSDK sdk("MacOSX.sdk");
  sdk.Merge(XcodeSDK("WatchOS.sdk"));
  // This doesn't make any particular sense and shouldn't happen in practice, we
  // just want to guarantee a well-defined behavior when choosing one
  // SDK to fit all CUs in an lldb::Module.
  // -> The higher number wins.
  EXPECT_EQ(sdk.GetType(), XcodeSDK::watchOS);
  sdk.Merge(XcodeSDK("WatchOS1.1.sdk"));
  EXPECT_EQ(sdk.GetVersion(), llvm::VersionTuple(1, 1));
  sdk.Merge(XcodeSDK("WatchOS2.0.sdk"));
  EXPECT_EQ(sdk.GetVersion(), llvm::VersionTuple(2, 0));
  sdk.Merge(XcodeSDK("WatchOS1.1.Internal.sdk"));
  EXPECT_EQ(sdk.GetVersion(), llvm::VersionTuple(2, 0));
  EXPECT_EQ(sdk.IsAppleInternalSDK(), true);
  XcodeSDK empty;
  empty.Merge(XcodeSDK("MacOSX10.14.Internal.sdk"));
  EXPECT_EQ(empty.GetString(), llvm::StringRef("MacOSX10.14.Internal.sdk"));
}

#ifndef _WIN32
TEST(XcodeSDKTest, SDKSupportsModules) {
  std::string base = "/Applications/Xcode.app/Contents/Developer/Platforms/";
  EXPECT_TRUE(XcodeSDK::SDKSupportsModules(
      XcodeSDK::Type::iPhoneSimulator,
      FileSpec(
          base +
          "iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator12.0.sdk")));
  EXPECT_TRUE(XcodeSDK::SDKSupportsModules(
      XcodeSDK::Type::iPhoneSimulator,
      FileSpec(base + "iPhoneSimulator.platform/Developer/SDKs/"
                      "iPhoneSimulator12.0.Internal.sdk")));
  EXPECT_FALSE(XcodeSDK::SDKSupportsModules(
      XcodeSDK::Type::iPhoneSimulator,
      FileSpec(
          base +
          "iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator7.2.sdk")));
  EXPECT_TRUE(XcodeSDK::SDKSupportsModules(
      XcodeSDK::Type::MacOSX,
      FileSpec(base + "MacOSX.platform/Developer/SDKs/MacOSX10.10.sdk")));
  EXPECT_FALSE(XcodeSDK::SDKSupportsModules(
      XcodeSDK::Type::MacOSX,
      FileSpec(base + "MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk")));
}
#endif

TEST(XcodeSDKTest, SDKSupportsSwift) {
  EXPECT_TRUE(XcodeSDK("iPhoneSimulator12.0.sdk").SupportsSwift());
  EXPECT_TRUE(XcodeSDK("iPhoneSimulator12.0.Internal.sdk").SupportsSwift());
  EXPECT_FALSE(XcodeSDK("iPhoneSimulator7.2.sdk").SupportsSwift());
  EXPECT_TRUE(XcodeSDK("MacOSX10.10.sdk").SupportsSwift());
  EXPECT_FALSE(XcodeSDK("MacOSX10.9.sdk").SupportsSwift());
  EXPECT_TRUE(XcodeSDK("Linux.sdk").SupportsSwift());
  EXPECT_TRUE(XcodeSDK("MacOSX.sdk").SupportsSwift());
  EXPECT_FALSE(XcodeSDK("EverythingElse.sdk").SupportsSwift());
}

TEST(XcodeSDKTest, GetCanonicalNameAndConstruct) {
  XcodeSDK::Info info;
  info.type = XcodeSDK::Type::MacOSX;
  EXPECT_EQ("macosx", XcodeSDK::GetCanonicalName(info));
  EXPECT_EQ(XcodeSDK(info).Parse(), info);

  info.type = XcodeSDK::Type::iPhoneSimulator;
  EXPECT_EQ("iphonesimulator", XcodeSDK::GetCanonicalName(info));
  EXPECT_EQ(XcodeSDK(info).Parse(), info);

  info.type = XcodeSDK::Type::iPhoneOS;
  EXPECT_EQ("iphoneos", XcodeSDK::GetCanonicalName(info));
  EXPECT_EQ(XcodeSDK(info).Parse(), info);

  info.type = XcodeSDK::Type::AppleTVSimulator;
  EXPECT_EQ("appletvsimulator", XcodeSDK::GetCanonicalName(info));
  EXPECT_EQ(XcodeSDK(info).Parse(), info);

  info.type = XcodeSDK::Type::AppleTVOS;
  EXPECT_EQ("appletvos", XcodeSDK::GetCanonicalName(info));
  EXPECT_EQ(XcodeSDK(info).Parse(), info);

  info.type = XcodeSDK::Type::WatchSimulator;
  EXPECT_EQ("watchsimulator", XcodeSDK::GetCanonicalName(info));
  EXPECT_EQ(XcodeSDK(info).Parse(), info);

  info.type = XcodeSDK::Type::watchOS;
  EXPECT_EQ("watchos", XcodeSDK::GetCanonicalName(info));
  EXPECT_EQ(XcodeSDK(info).Parse(), info);

  info.type = XcodeSDK::Type::Linux;
  EXPECT_EQ("linux", XcodeSDK::GetCanonicalName(info));
  EXPECT_EQ(XcodeSDK(info).Parse(), info);

  info.type = XcodeSDK::Type::unknown;
  EXPECT_EQ("", XcodeSDK::GetCanonicalName(info));
  EXPECT_EQ(XcodeSDK(info).Parse(), info);

  info.internal = true;
  info.type = XcodeSDK::Type::MacOSX;
  EXPECT_EQ("macosx.internal", XcodeSDK::GetCanonicalName(info));
  EXPECT_EQ(XcodeSDK(info).Parse(), info);

  info.type = XcodeSDK::Type::iPhoneSimulator;
  EXPECT_EQ("iphonesimulator.internal", XcodeSDK::GetCanonicalName(info));
  EXPECT_EQ(XcodeSDK(info).Parse(), info);

  info.type = XcodeSDK::Type::iPhoneOS;
  EXPECT_EQ("iphoneos.internal", XcodeSDK::GetCanonicalName(info));
  EXPECT_EQ(XcodeSDK(info).Parse(), info);

  info.type = XcodeSDK::Type::AppleTVSimulator;
  EXPECT_EQ("appletvsimulator.internal", XcodeSDK::GetCanonicalName(info));
  EXPECT_EQ(XcodeSDK(info).Parse(), info);

  info.type = XcodeSDK::Type::AppleTVOS;
  EXPECT_EQ("appletvos.internal", XcodeSDK::GetCanonicalName(info));
  EXPECT_EQ(XcodeSDK(info).Parse(), info);

  info.type = XcodeSDK::Type::WatchSimulator;
  EXPECT_EQ("watchsimulator.internal", XcodeSDK::GetCanonicalName(info));
  EXPECT_EQ(XcodeSDK(info).Parse(), info);

  info.type = XcodeSDK::Type::watchOS;
  EXPECT_EQ("watchos.internal", XcodeSDK::GetCanonicalName(info));
  EXPECT_EQ(XcodeSDK(info).Parse(), info);

  info.type = XcodeSDK::Type::MacOSX;
  info.version = llvm::VersionTuple(10, 9);
  EXPECT_EQ("macosx10.9.internal", XcodeSDK::GetCanonicalName(info));
  EXPECT_EQ(XcodeSDK(info).Parse(), info);

  info.type = XcodeSDK::Type::iPhoneOS;
  info.version = llvm::VersionTuple(7, 0);
  EXPECT_EQ("iphoneos7.0.internal", XcodeSDK::GetCanonicalName(info));
  EXPECT_EQ(XcodeSDK(info).Parse(), info);
}

TEST(XcodeSDKTest, GetSDKTypeForTriple) {
  EXPECT_EQ(
      XcodeSDK::GetSDKTypeForTriple(llvm::Triple("x86_64-apple-macosx10.14")),
      XcodeSDK::Type::MacOSX);
  EXPECT_EQ(XcodeSDK::GetSDKTypeForTriple(llvm::Triple("x86_64-apple-darwin")),
            XcodeSDK::Type::MacOSX);
  EXPECT_EQ(XcodeSDK::GetSDKTypeForTriple(
                llvm::Triple("x86_64-apple-ios13.4-simulator")),
            XcodeSDK::Type::iPhoneSimulator);
  EXPECT_EQ(XcodeSDK::GetSDKTypeForTriple(llvm::Triple("arm64-apple-ios13.4")),
            XcodeSDK::Type::iPhoneOS);
  EXPECT_EQ(XcodeSDK::GetSDKTypeForTriple(
                llvm::Triple("x86_64-apple-ios13.4-macabi")),
            XcodeSDK::Type::MacOSX);
  EXPECT_EQ(XcodeSDK::GetSDKTypeForTriple(
                llvm::Triple("x86_64-apple-tvos-simulator")),
            XcodeSDK::Type::AppleTVSimulator);
  EXPECT_EQ(XcodeSDK::GetSDKTypeForTriple(llvm::Triple("arm64-apple-tvos")),
            XcodeSDK::Type::AppleTVOS);
  EXPECT_EQ(XcodeSDK::GetSDKTypeForTriple(
                llvm::Triple("x86_64-apple-watchos-simulator")),
            XcodeSDK::Type::WatchSimulator);
  EXPECT_EQ(XcodeSDK::GetSDKTypeForTriple(llvm::Triple("arm64-apple-watchos")),
            XcodeSDK::Type::watchOS);
  EXPECT_EQ(XcodeSDK::GetSDKTypeForTriple(llvm::Triple("x86_64-unknown-linux")),
            XcodeSDK::Type::Linux);
  EXPECT_EQ(XcodeSDK::GetSDKTypeForTriple(llvm::Triple("i386-unknown-netbsd")),
            XcodeSDK::Type::unknown);
}

TEST(XcodeSDKTest, FindXcodeContentsDirectoryInPath) {
  std::string standard =
      "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/"
      "Developer/SDKs/MacOSX.sdk";
  EXPECT_EQ("/Applications/Xcode.app/Contents",
            XcodeSDK::FindXcodeContentsDirectoryInPath(standard));

  std::string standard_version =
      "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/"
      "Developer/SDKs/MacOSX10.15.sdk";
  EXPECT_EQ("/Applications/Xcode.app/Contents",
            XcodeSDK::FindXcodeContentsDirectoryInPath(standard_version));

  std::string beta = "/Applications/Xcode-beta.app/Contents/Developer/"
                     "Platforms/MacOSX.platform/"
                     "Developer/SDKs/MacOSX10.15.sdk";
  EXPECT_EQ("/Applications/Xcode-beta.app/Contents",
            XcodeSDK::FindXcodeContentsDirectoryInPath(beta));

  std::string no_app =
      "/Applications/Xcode/Contents/Developer/Platforms/MacOSX.platform/"
      "Developer/SDKs/MacOSX10.15.sdk";
  EXPECT_EQ("", XcodeSDK::FindXcodeContentsDirectoryInPath(no_app));

  std::string no_contents =
      "/Applications/Xcode.app/Developer/Platforms/MacOSX.platform/"
      "Developer/SDKs/MacOSX10.15.sdk";
  EXPECT_EQ("", XcodeSDK::FindXcodeContentsDirectoryInPath(no_contents));

  std::string no_capitalization =
      "/Applications/Xcode.app/contents/Developer/Platforms/MacOSX.platform/"
      "Developer/SDKs/MacOSX10.15.sdk";
  EXPECT_EQ("", XcodeSDK::FindXcodeContentsDirectoryInPath(no_capitalization));
}
