//===-- EnvironmentTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Utility/Environment.h"

using namespace lldb_private;

TEST(EnvironmentTest, EnvpConstruction) {
  const char **Envp1 = nullptr;
  EXPECT_EQ(0u, Environment(Envp1).size());

  const char *Envp2[] = {"FOO=BAR", nullptr};
  EXPECT_EQ("BAR", Environment(Envp2).lookup("FOO"));

  const char *Envp3[] = {"FOO=BAR", "FOO=BAZ", nullptr};
  EXPECT_EQ("BAR", Environment(Envp3).lookup("FOO"));

  const char *Envp4[] = {"FOO=", "BAR", nullptr};
  Environment Env4(Envp4);
  ASSERT_EQ(2u, Env4.size());
  EXPECT_EQ("", Environment(Envp4).find("FOO")->second);
  EXPECT_EQ("", Environment(Envp4).find("BAR")->second);

  const char *Envp5[] = {"FOO=BAR=BAZ", nullptr};
  EXPECT_EQ("BAR=BAZ", Environment(Envp5).lookup("FOO"));
}

TEST(EnvironmentTest, EnvpConversion) {
  std::string FOO_EQ_BAR("FOO=BAR");
  std::string BAR_EQ_BAZ("BAR=BAZ");

  Environment Env;
  Env.insert(FOO_EQ_BAR);
  Env.insert(BAR_EQ_BAZ);
  Environment::Envp Envp = Env.getEnvp();
  const char *const *Envp_ = Envp;

  EXPECT_TRUE(FOO_EQ_BAR == Envp_[0] || FOO_EQ_BAR == Envp_[1]);
  EXPECT_TRUE(BAR_EQ_BAZ == Envp_[0] || BAR_EQ_BAZ == Envp_[1]);
  EXPECT_EQ(nullptr, Envp_[2]);
}
