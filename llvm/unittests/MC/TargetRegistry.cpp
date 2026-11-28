//===- unittests/MC/TargetRegistry.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The target registry code lives in Support, but it relies on linking in all
// LLVM targets. We keep this test with the MC tests, which already do that, to
// keep the SupportTests target small.

#include "llvm/MC/TargetRegistry.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"
#if LLVM_ENABLE_THREADS
#include <array>
#include <atomic>
#include <thread>
#include <vector>
#endif

using namespace llvm;

namespace {

TEST(TargetRegistry, TargetHasArchType) {
  // Presence of at least one target will be asserted when done with the loop,
  // else this would pass by accident if InitializeAllTargetInfos were omitted.
  int Count = 0;

  llvm::InitializeAllTargetInfos();

  for (const Target &T : TargetRegistry::targets()) {
    StringRef Name = T.getName();
    // There is really no way (at present) to ask a Target whether it targets
    // a specific architecture, because the logic for that is buried in a
    // predicate.
    // We can't ask the predicate "Are you a function that always returns
    // false?"
    // So given that the cpp backend truly has no target arch, it is skipped.
    if (Name != "cpp") {
      Triple::ArchType Arch = Triple::getArchTypeForLLVMName(Name);
      EXPECT_NE(Arch, Triple::UnknownArch);
      ++Count;
    }
  }
  ASSERT_NE(Count, 0);
}

TEST(TargetRegistry, IsValidFeatureListFormat) {
  // Valid strings

  // Empty string is a valid feature string
  EXPECT_TRUE(Target::isValidFeatureListFormat(""));

  EXPECT_TRUE(Target::isValidFeatureListFormat("+some_feature"));
  EXPECT_TRUE(Target::isValidFeatureListFormat("-some_feature"));
  EXPECT_TRUE(
      Target::isValidFeatureListFormat("+feature1,-feature2,+feature3"));
  EXPECT_TRUE(Target::isValidFeatureListFormat("+123"));

  // Strings with single trailing comma are also valid
  EXPECT_TRUE(Target::isValidFeatureListFormat("-feature,"));
  EXPECT_TRUE(
      Target::isValidFeatureListFormat("-feature1,+feature2,+feature3,"));

  // Invalid strings

  // Feature don't start with '+' or '-'
  EXPECT_FALSE(Target::isValidFeatureListFormat("invalid_string"));
  EXPECT_FALSE(Target::isValidFeatureListFormat("+good,bad"));
  EXPECT_FALSE(Target::isValidFeatureListFormat("bad,+good"));

  // String has spaces
  EXPECT_FALSE(Target::isValidFeatureListFormat(" "));
  EXPECT_FALSE(Target::isValidFeatureListFormat(", "));
  EXPECT_FALSE(Target::isValidFeatureListFormat(" avx"));
  EXPECT_FALSE(Target::isValidFeatureListFormat("+avx, -sse"));

  // Redundant commas
  EXPECT_FALSE(Target::isValidFeatureListFormat("+feature1,,+feature2"));
  EXPECT_FALSE(Target::isValidFeatureListFormat(",+feature"));
  EXPECT_FALSE(
      Target::isValidFeatureListFormat("+feature1,,,+feature2,,+feature3"));

  // Feature consists only of '+' or '-'
  EXPECT_FALSE(Target::isValidFeatureListFormat("+"));
  EXPECT_FALSE(Target::isValidFeatureListFormat("-"));
  EXPECT_FALSE(Target::isValidFeatureListFormat("+avx,-"));

  // Only commas
  EXPECT_FALSE(Target::isValidFeatureListFormat(","));
  EXPECT_FALSE(Target::isValidFeatureListFormat(",,"));
  EXPECT_FALSE(Target::isValidFeatureListFormat(",,,"));
}

TEST(TargetRegistry, SubtargetCopyPreservesHwMode) {
  InitializeAllTargetInfos();
  InitializeAllTargetMCs();

  for (StringRef TripleName :
       {"x86_64-unknown-linux-gnu", "riscv64-unknown-elf"}) {
    SCOPED_TRACE(TripleName);
    Triple TT(TripleName);
    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);
    if (!TheTarget)
      continue;
    std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TT));
    MCTargetOptions MCOptions;
    std::unique_ptr<MCAsmInfo> MAI(
        TheTarget->createMCAsmInfo(*MRI, TT, MCOptions));
    std::unique_ptr<MCSubtargetInfo> STI(
        TheTarget->createMCSubtargetInfo(TT, "", ""));
    ASSERT_TRUE(MRI && MAI && STI);
    ASSERT_NE(STI->getHwMode(), 0u);
    ASSERT_NE(STI->getHwModeSet(), 0u);
    MCContext Ctx(TT, *MAI, *MRI, *STI);
    MCSubtargetInfo &Copy = Ctx.getSubtargetCopy(*STI);
    EXPECT_EQ(Copy.getHwMode(), STI->getHwMode());
    EXPECT_EQ(Copy.getHwModeSet(), STI->getHwModeSet());
  }
}

#if LLVM_ENABLE_THREADS
TEST(TargetRegistry, ConcurrentRegisterAndLookup) {
  constexpr size_t NumTargets = 128;
  constexpr int NumWriters = 4;
  constexpr int NumReaders = 4;

  // Ensure at least one real target is already registered so readers verify
  // that looking up an already-registered target succeeds while other targets
  // are concurrently registered.
  InitializeAllTargetInfos();
  const Target *ExistingTarget = nullptr;
  Triple ExistingTriple;
  for (const Target &T : TargetRegistry::targets()) {
    StringRef Name = T.getName();
    if (Name == "cpp")
      continue;
    Triple::ArchType Arch = Triple::getArchTypeForLLVMName(Name);
    if (Arch == Triple::UnknownArch)
      continue;
    Triple Candidate;
    Candidate.setArch(Arch);
    std::string InitError;
    if (TargetRegistry::lookupTarget(Candidate, InitError) == &T) {
      ExistingTarget = &T;
      ExistingTriple = Candidate;
      break;
    }
  }
  ASSERT_NE(ExistingTarget, nullptr);

  // TargetRegistry has no unregister API, so keep test Targets alive for the
  // process lifetime. Use Name="cpp" (skipped by TargetHasArchType) and a
  // non-matching ArchMatchFn so other tests in this binary are unaffected.
  static std::array<Target, NumTargets> Targets;
  auto NeverMatch = [](Triple::ArchType) { return false; };

  std::atomic<int> Ready{0};
  std::atomic<bool> Start{false};
  std::atomic<int> ActiveWriters{NumWriters};
  std::atomic<int> ReadersCompleted{0};

  std::vector<std::thread> Writers;
  Writers.reserve(NumWriters);
  for (int W = 0; W < NumWriters; ++W) {
    Writers.emplace_back([&]() {
      Ready.fetch_add(1, std::memory_order_release);
      while (!Start.load(std::memory_order_acquire))
        std::this_thread::yield();
      do {
        for (size_t I = 0; I < NumTargets; ++I) {
          TargetRegistry::RegisterTarget(Targets[I], "cpp", "test target",
                                         "test", NeverMatch);
        }
      } while (ReadersCompleted.load(std::memory_order_acquire) < NumReaders);
      ActiveWriters.fetch_sub(1, std::memory_order_release);
    });
  }

  std::vector<std::thread> Readers;
  Readers.reserve(NumReaders);
  for (int R = 0; R < NumReaders; ++R) {
    Readers.emplace_back([&]() {
      Ready.fetch_add(1, std::memory_order_release);
      while (!Start.load(std::memory_order_acquire))
        std::this_thread::yield();
      bool CompletedPass = false;
      while (!CompletedPass ||
             ActiveWriters.load(std::memory_order_acquire) > 0) {
        std::string Error;
        EXPECT_EQ(TargetRegistry::lookupTarget(ExistingTriple, Error),
                  ExistingTarget);
        Triple LookupTriple = ExistingTriple;
        EXPECT_EQ(TargetRegistry::lookupTarget(ExistingTarget->getName(),
                                               LookupTriple, Error),
                  ExistingTarget);

        bool SawTestTarget = false;
        DenseSet<const Target *> Seen;
        for (const Target &T : TargetRegistry::targets()) {
          if (!Seen.insert(&T).second) {
            ADD_FAILURE() << "Cycle detected in TargetRegistry list";
            break;
          }
          const char *Name = T.getName();
          EXPECT_NE(Name, nullptr);
          EXPECT_NE(T.getShortDescription(), nullptr);
          EXPECT_NE(T.getBackendName(), nullptr);
          if (Name && StringRef(Name) == "cpp")
            SawTestTarget = true;
        }
        if (!CompletedPass && SawTestTarget) {
          CompletedPass = true;
          ReadersCompleted.fetch_add(1, std::memory_order_release);
        }
      }
    });
  }

  while (Ready.load(std::memory_order_acquire) < NumWriters + NumReaders)
    std::this_thread::yield();
  Start.store(true, std::memory_order_release);
  for (std::thread &W : Writers)
    W.join();
  for (std::thread &R : Readers)
    R.join();

  EXPECT_EQ(ReadersCompleted.load(std::memory_order_relaxed), NumReaders);

  DenseSet<const Target *> Seen;
  for (const Target &T : TargetRegistry::targets())
    ASSERT_TRUE(Seen.insert(&T).second) << "Duplicate or cycle in registry";

  for (size_t I = 0; I < NumTargets; ++I) {
    EXPECT_EQ(Seen.count(&Targets[I]), 1u);
    EXPECT_STREQ(Targets[I].getName(), "cpp");
    EXPECT_STREQ(Targets[I].getShortDescription(), "test target");
    EXPECT_STREQ(Targets[I].getBackendName(), "test");
  }
}
#endif

} // end namespace
