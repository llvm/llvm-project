//===-- ReproducerTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Reproducer.h"
#include "lldb/Utility/ReproducerProvider.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Error.h"

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

class YamlMultiProvider
    : public MultiProvider<YamlRecorder, YamlMultiProvider> {
public:
  struct Info {
    static const char *name;
    static const char *file;
  };

  YamlMultiProvider(const FileSpec &directory) : MultiProvider(directory) {}

  static char ID;
};

const char *DummyProvider::Info::name = "dummy";
const char *DummyProvider::Info::file = "dummy.yaml";
const char *YamlMultiProvider::Info::name = "mutli";
const char *YamlMultiProvider::Info::file = "mutli.yaml";
char DummyProvider::ID = 0;
char YamlMultiProvider::ID = 0;

class DummyReproducer : public Reproducer {
public:
  DummyReproducer() : Reproducer(){};

  using Reproducer::SetCapture;
  using Reproducer::SetReplay;
};

struct YamlData {
  YamlData() : i(-1) {}
  YamlData(int i) : i(i) {}
  int i;
};

inline bool operator==(const YamlData &LHS, const YamlData &RHS) {
  return LHS.i == RHS.i;
}

LLVM_YAML_IS_DOCUMENT_LIST_VECTOR(YamlData)

namespace llvm {
namespace yaml {
template <> struct MappingTraits<YamlData> {
  static void mapping(IO &io, YamlData &Y) { io.mapRequired("i", Y.i); };
};
} // namespace yaml
} // namespace llvm

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

TEST(GeneratorTest, YamlMultiProvider) {
  SmallString<128> root;
  std::error_code ec = llvm::sys::fs::createUniqueDirectory("reproducer", root);
  ASSERT_FALSE(static_cast<bool>(ec));

  auto cleanup = llvm::make_scope_exit(
      [&] { EXPECT_FALSE(llvm::sys::fs::remove_directories(root.str())); });

  YamlData data0(0);
  YamlData data1(1);
  YamlData data2(2);
  YamlData data3(3);

  {
    DummyReproducer reproducer;
    EXPECT_THAT_ERROR(reproducer.SetCapture(FileSpec(root.str())), Succeeded());

    auto &generator = *reproducer.GetGenerator();
    auto *provider = generator.Create<YamlMultiProvider>();
    ASSERT_NE(nullptr, provider);

    auto *recorder = provider->GetNewRecorder();
    ASSERT_NE(nullptr, recorder);
    recorder->Record(data0);
    recorder->Record(data1);

    recorder = provider->GetNewRecorder();
    ASSERT_NE(nullptr, recorder);
    recorder->Record(data2);
    recorder->Record(data3);

    generator.Keep();
  }

  {
    DummyReproducer reproducer;
    EXPECT_THAT_ERROR(reproducer.SetReplay(FileSpec(root.str())), Succeeded());

    auto &loader = *reproducer.GetLoader();
    std::unique_ptr<repro::MultiLoader<YamlMultiProvider>> multi_loader =
        repro::MultiLoader<YamlMultiProvider>::Create(&loader);

    // Read the first file.
    {
      llvm::Optional<std::string> file = multi_loader->GetNextFile();
      EXPECT_TRUE(static_cast<bool>(file));

      auto buffer = llvm::MemoryBuffer::getFile(*file);
      EXPECT_TRUE(static_cast<bool>(buffer));

      yaml::Input yin((*buffer)->getBuffer());
      std::vector<YamlData> data;
      yin >> data;

      ASSERT_EQ(data.size(), 2U);
      EXPECT_THAT(data, testing::ElementsAre(data0, data1));
    }

    // Read the second file.
    {
      llvm::Optional<std::string> file = multi_loader->GetNextFile();
      EXPECT_TRUE(static_cast<bool>(file));

      auto buffer = llvm::MemoryBuffer::getFile(*file);
      EXPECT_TRUE(static_cast<bool>(buffer));

      yaml::Input yin((*buffer)->getBuffer());
      std::vector<YamlData> data;
      yin >> data;

      ASSERT_EQ(data.size(), 2U);
      EXPECT_THAT(data, testing::ElementsAre(data2, data3));
    }

    // There is no third file.
    llvm::Optional<std::string> file = multi_loader->GetNextFile();
    EXPECT_FALSE(static_cast<bool>(file));
  }
}
