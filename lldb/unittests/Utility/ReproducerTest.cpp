//===-- ReproducerTest.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  static constexpr const char *NAME = "dummy";

  DummyProvider(const FileSpec &directory) : Provider(directory) {
    m_info.name = "dummy";
    m_info.files.push_back("dummy.yaml");
  }

  static char ID;
};

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
  EXPECT_THAT_ERROR(reproducer.SetCapture(FileSpec("/bogus/path")),
                    Succeeded());
  EXPECT_NE(nullptr, reproducer.GetGenerator());
  EXPECT_EQ(FileSpec("/bogus/path"), reproducer.GetGenerator()->GetRoot());
  EXPECT_EQ(FileSpec("/bogus/path"), reproducer.GetReproducerPath());

  // Ensure that we cannot enable replay.
  EXPECT_THAT_ERROR(reproducer.SetReplay(FileSpec("/bogus/path")), Failed());
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
  EXPECT_THAT_ERROR(reproducer.SetReplay(FileSpec("/bogus/path")), Failed());
  // However the loader should still be set, which we check here.
  EXPECT_NE(nullptr, reproducer.GetLoader());

  // Make sure the bogus path is correctly set.
  EXPECT_EQ(FileSpec("/bogus/path"), reproducer.GetLoader()->GetRoot());
  EXPECT_EQ(FileSpec("/bogus/path"), reproducer.GetReproducerPath());

  // Ensure that we cannot enable replay.
  EXPECT_THAT_ERROR(reproducer.SetCapture(FileSpec("/bogus/path")), Failed());
  EXPECT_EQ(nullptr, reproducer.GetGenerator());
}

TEST(GeneratorTest, Create) {
  DummyReproducer reproducer;

  EXPECT_THAT_ERROR(reproducer.SetCapture(FileSpec("/bogus/path")),
                    Succeeded());
  auto &generator = *reproducer.GetGenerator();

  auto *provider = generator.Create<DummyProvider>();
  EXPECT_NE(nullptr, provider);
  EXPECT_EQ(FileSpec("/bogus/path"), provider->GetRoot());
  EXPECT_EQ(std::string("dummy"), provider->GetInfo().name);
  EXPECT_EQ((size_t)1, provider->GetInfo().files.size());
  EXPECT_EQ(std::string("dummy.yaml"), provider->GetInfo().files.front());
}

TEST(GeneratorTest, Get) {
  DummyReproducer reproducer;

  EXPECT_THAT_ERROR(reproducer.SetCapture(FileSpec("/bogus/path")),
                    Succeeded());
  auto &generator = *reproducer.GetGenerator();

  auto *provider = generator.Create<DummyProvider>();
  EXPECT_NE(nullptr, provider);

  auto *provider_alt = generator.Get<DummyProvider>();
  EXPECT_EQ(provider, provider_alt);
}

TEST(GeneratorTest, GetOrCreate) {
  DummyReproducer reproducer;

  EXPECT_THAT_ERROR(reproducer.SetCapture(FileSpec("/bogus/path")),
                    Succeeded());
  auto &generator = *reproducer.GetGenerator();

  auto &provider = generator.GetOrCreate<DummyProvider>();
  EXPECT_EQ(FileSpec("/bogus/path"), provider.GetRoot());
  EXPECT_EQ(std::string("dummy"), provider.GetInfo().name);
  EXPECT_EQ((size_t)1, provider.GetInfo().files.size());
  EXPECT_EQ(std::string("dummy.yaml"), provider.GetInfo().files.front());

  auto &provider_alt = generator.GetOrCreate<DummyProvider>();
  EXPECT_EQ(&provider, &provider_alt);
}
