//===-- ReproducerTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Error.h"

#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Reproducer.h"

using namespace llvm;
using namespace lldb_private;
using namespace lldb_private::repro;

class DummyProvider : public repro::Provider<DummyProvider> {
public:
  struct Info {
    static const char *name;
    static const char *file;
  };

  DummyProvider(const FileSpec &directory) : Provider(directory) {}

  static char ID;
};

const char *DummyProvider::Info::name = "dummy";
const char *DummyProvider::Info::file = "dummy.yaml";

class DummyReproducer : public Reproducer {
public:
  DummyReproducer() : Reproducer(){};

  using Reproducer::SetCapture;
  using Reproducer::SetReplay;
};

char DummyProvider::ID = 0;

TEST(ReproducerTest, SetCapture) {
  DummyReproducer reproducer;

  // Initially both generator and loader are unset.
  EXPECT_EQ(nullptr, reproducer.GetGenerator());
  EXPECT_EQ(nullptr, reproducer.GetLoader());

  // Enable capture and check that means we have a generator.
  EXPECT_THAT_ERROR(
      reproducer.SetCapture(FileSpec("//bogus/path", FileSpec::Style::posix)),
      Succeeded());
  EXPECT_NE(nullptr, reproducer.GetGenerator());
  EXPECT_EQ(FileSpec("//bogus/path", FileSpec::Style::posix),
            reproducer.GetGenerator()->GetRoot());
  EXPECT_EQ(FileSpec("//bogus/path", FileSpec::Style::posix),
            reproducer.GetReproducerPath());

  // Ensure that we cannot enable replay.
  EXPECT_THAT_ERROR(
      reproducer.SetReplay(FileSpec("//bogus/path", FileSpec::Style::posix)),
      Failed());
  EXPECT_EQ(nullptr, reproducer.GetLoader());

  // Ensure we can disable the generator again.
  EXPECT_THAT_ERROR(reproducer.SetCapture(llvm::None), Succeeded());
  EXPECT_EQ(nullptr, reproducer.GetGenerator());
  EXPECT_EQ(nullptr, reproducer.GetLoader());
}

TEST(ReproducerTest, SetReplay) {
  DummyReproducer reproducer;

  // Initially both generator and loader are unset.
  EXPECT_EQ(nullptr, reproducer.GetGenerator());
  EXPECT_EQ(nullptr, reproducer.GetLoader());

  // Expected to fail because we can't load the index.
  EXPECT_THAT_ERROR(
      reproducer.SetReplay(FileSpec("//bogus/path", FileSpec::Style::posix)),
      Failed());
  // However the loader should still be set, which we check here.
  EXPECT_NE(nullptr, reproducer.GetLoader());

  // Make sure the bogus path is correctly set.
  EXPECT_EQ(FileSpec("//bogus/path", FileSpec::Style::posix),
            reproducer.GetLoader()->GetRoot());
  EXPECT_EQ(FileSpec("//bogus/path", FileSpec::Style::posix),
            reproducer.GetReproducerPath());

  // Ensure that we cannot enable replay.
  EXPECT_THAT_ERROR(
      reproducer.SetCapture(FileSpec("//bogus/path", FileSpec::Style::posix)),
      Failed());
  EXPECT_EQ(nullptr, reproducer.GetGenerator());
}

TEST(GeneratorTest, Create) {
  DummyReproducer reproducer;

  EXPECT_THAT_ERROR(
      reproducer.SetCapture(FileSpec("//bogus/path", FileSpec::Style::posix)),
      Succeeded());
  auto &generator = *reproducer.GetGenerator();

  auto *provider = generator.Create<DummyProvider>();
  EXPECT_NE(nullptr, provider);
  EXPECT_EQ(FileSpec("//bogus/path", FileSpec::Style::posix),
            provider->GetRoot());
}

TEST(GeneratorTest, Get) {
  DummyReproducer reproducer;

  EXPECT_THAT_ERROR(
      reproducer.SetCapture(FileSpec("//bogus/path", FileSpec::Style::posix)),
      Succeeded());
  auto &generator = *reproducer.GetGenerator();

  auto *provider = generator.Create<DummyProvider>();
  EXPECT_NE(nullptr, provider);

  auto *provider_alt = generator.Get<DummyProvider>();
  EXPECT_EQ(provider, provider_alt);
}

TEST(GeneratorTest, GetOrCreate) {
  DummyReproducer reproducer;

  EXPECT_THAT_ERROR(
      reproducer.SetCapture(FileSpec("//bogus/path", FileSpec::Style::posix)),
      Succeeded());
  auto &generator = *reproducer.GetGenerator();

  auto &provider = generator.GetOrCreate<DummyProvider>();
  EXPECT_EQ(FileSpec("//bogus/path", FileSpec::Style::posix),
            provider.GetRoot());

  auto &provider_alt = generator.GetOrCreate<DummyProvider>();
  EXPECT_EQ(&provider, &provider_alt);
}
