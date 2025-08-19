//===- CASConfiguration.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/CASConfiguration.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::cas;

TEST(CASConfigurationTest, roundTrips) {
  auto roundTripConfig = [](CASConfiguration &Config) {
    std::string Serialized;
    raw_string_ostream OS(Serialized);
    Config.writeConfigurationFile(OS);

    std::optional<CASConfiguration> NewConfig;
    ASSERT_THAT_ERROR(
        CASConfiguration::createFromConfig(Serialized).moveInto(NewConfig),
        Succeeded());
    ASSERT_TRUE(Config == *NewConfig);
  };

  CASConfiguration Config;
  roundTripConfig(Config);

  Config.CASPath = "/tmp";
  roundTripConfig(Config);

  Config.PluginPath = "/test.plug";
  roundTripConfig(Config);

  Config.PluginOptions.emplace_back("a", "b");
  roundTripConfig(Config);

  Config.PluginOptions.emplace_back("c", "d");
  roundTripConfig(Config);
}

TEST(CASConfigurationTest, configFileSearch) {
  // /a/b/c
  SmallString<261> PathToC = sys::path::get_separator();
  sys::path::append(PathToC, "a", "b", "c");
  // /a/b/c/d
  SmallString<261> PathToD = PathToC;
  sys::path::append(PathToD, "d");
  // /a/b/c/d/e
  SmallString<261> PathToE = PathToD;
  sys::path::append(PathToE, "e");
  // /a/b/c/.cas-config
  SmallString<261> PathToCConfig = PathToC;
  sys::path::append(PathToCConfig, ".cas-config");
  // /a/b/c/d/.cas-config
  SmallString<261> PathToDConfig = PathToD;
  sys::path::append(PathToDConfig, ".cas-config");

  auto VFS = makeIntrusiveRefCnt<vfs::InMemoryFileSystem>();
  ASSERT_FALSE(
      CASConfiguration::createFromSearchConfigFile(PathToE.str(), VFS));

  // Add an empty file.
  VFS->addFile(PathToCConfig.str(), 0,
               llvm::MemoryBuffer::getMemBufferCopy(""));
  ASSERT_FALSE(
      CASConfiguration::createFromSearchConfigFile(PathToE.str(), VFS));

  VFS->addFile(PathToDConfig.str(), 0,
#ifndef _WIN32
      llvm::MemoryBuffer::getMemBufferCopy("{\"CASPath\": \"/tmp\"}"));  
#else
      llvm::MemoryBuffer::getMemBufferCopy("{\"CASPath\": \"\\\\tmp\"}"));
#endif

  CASConfiguration Config;
  SmallString<261> CASPath = sys::path::get_separator();
  sys::path::append(CASPath, "tmp");
  Config.CASPath = CASPath.str();
  auto NewConfig =
      CASConfiguration::createFromSearchConfigFile(PathToE.str(), VFS);
  ASSERT_TRUE(NewConfig);
  ASSERT_TRUE(NewConfig->first == PathToDConfig.str());
  ASSERT_TRUE(Config == NewConfig->second);
}
