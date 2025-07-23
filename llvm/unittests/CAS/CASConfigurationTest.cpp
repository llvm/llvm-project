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
  auto VFS = makeIntrusiveRefCnt<vfs::InMemoryFileSystem>();
  ASSERT_FALSE(CASConfiguration::createFromSearchConfigFile("/a/b/c/d/e", VFS));

  // Add an empty file.
  VFS->addFile("/a/b/c/.cas-config", 0,
               llvm::MemoryBuffer::getMemBufferCopy(""));
  ASSERT_FALSE(CASConfiguration::createFromSearchConfigFile("/a/b/c/d/e", VFS));

  VFS->addFile("/a/b/c/d/.cas-config", 0,
               llvm::MemoryBuffer::getMemBufferCopy("{\"CASPath\": \"/tmp\"}"));
  CASConfiguration Config;
  Config.CASPath = "/tmp";
  auto NewConfig =
      CASConfiguration::createFromSearchConfigFile("/a/b/c/d/e", VFS);
  ASSERT_TRUE(NewConfig);
  ASSERT_TRUE(NewConfig->first == "/a/b/c/d/.cas-config");
  ASSERT_TRUE(Config == NewConfig->second);
}
