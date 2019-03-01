//===- unittest/ProfileData/SampleProfTest.cpp ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/SampleProf.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/ProfileData/SampleProfReader.h"
#include "llvm/ProfileData/SampleProfWriter.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <string>
#include <vector>

using namespace llvm;
using namespace sampleprof;

static ::testing::AssertionResult NoError(std::error_code EC) {
  if (!EC)
    return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << "error " << EC.value() << ": "
                                       << EC.message();
}

namespace {

struct SampleProfTest : ::testing::Test {
  LLVMContext Context;
  std::unique_ptr<SampleProfileWriter> Writer;
  std::unique_ptr<SampleProfileReader> Reader;

  SampleProfTest() : Writer(), Reader() {}

  void createWriter(SampleProfileFormat Format, StringRef Profile) {
    std::error_code EC;
    std::unique_ptr<raw_ostream> OS(
        new raw_fd_ostream(Profile, EC, sys::fs::F_None));
    auto WriterOrErr = SampleProfileWriter::create(OS, Format);
    ASSERT_TRUE(NoError(WriterOrErr.getError()));
    Writer = std::move(WriterOrErr.get());
  }

  void readProfile(const Module &M, StringRef Profile) {
    auto ReaderOrErr = SampleProfileReader::create(Profile, Context);
    ASSERT_TRUE(NoError(ReaderOrErr.getError()));
    Reader = std::move(ReaderOrErr.get());
    Reader->collectFuncsToUse(M);
  }

  void testRoundTrip(SampleProfileFormat Format, bool Remap) {
    SmallVector<char, 128> ProfilePath;
    ASSERT_TRUE(NoError(llvm::sys::fs::createTemporaryFile("profile", "", ProfilePath)));
    StringRef Profile(ProfilePath.data(), ProfilePath.size());
    createWriter(Format, Profile);

    StringRef FooName("_Z3fooi");
    FunctionSamples FooSamples;
    FooSamples.setName(FooName);
    FooSamples.addTotalSamples(7711);
    FooSamples.addHeadSamples(610);
    FooSamples.addBodySamples(1, 0, 610);
    FooSamples.addBodySamples(2, 0, 600);
    FooSamples.addBodySamples(4, 0, 60000);
    FooSamples.addBodySamples(8, 0, 60351);
    FooSamples.addBodySamples(10, 0, 605);

    StringRef BarName("_Z3bari");
    FunctionSamples BarSamples;
    BarSamples.setName(BarName);
    BarSamples.addTotalSamples(20301);
    BarSamples.addHeadSamples(1437);
    BarSamples.addBodySamples(1, 0, 1437);
    // Test how reader/writer handles unmangled names.
    StringRef MconstructName("_M_construct<char *>");
    StringRef StringviewName("string_view<std::allocator<char> >");
    BarSamples.addCalledTargetSamples(1, 0, MconstructName, 1000);
    BarSamples.addCalledTargetSamples(1, 0, StringviewName, 437);

    Module M("my_module", Context);
    FunctionType *fn_type =
        FunctionType::get(Type::getVoidTy(Context), {}, false);
    M.getOrInsertFunction(FooName, fn_type);
    M.getOrInsertFunction(BarName, fn_type);

    StringMap<FunctionSamples> Profiles;
    Profiles[FooName] = std::move(FooSamples);
    Profiles[BarName] = std::move(BarSamples);

    std::error_code EC;
    EC = Writer->write(Profiles);
    ASSERT_TRUE(NoError(EC));

    Writer->getOutputStream().flush();

    readProfile(M, Profile);

    EC = Reader->read();
    ASSERT_TRUE(NoError(EC));

    if (Remap) {
      auto MemBuffer = llvm::MemoryBuffer::getMemBuffer(R"(
        # Types 'int' and 'long' are equivalent
        type i l
        # Function names 'foo' and 'faux' are equivalent
        name 3foo 4faux
      )");
      Reader.reset(new SampleProfileReaderItaniumRemapper(
          std::move(MemBuffer), Context, std::move(Reader)));
      FooName = "_Z4fauxi";
      BarName = "_Z3barl";

      EC = Reader->read();
      ASSERT_TRUE(NoError(EC));
    }

    ASSERT_EQ(2u, Reader->getProfiles().size());

    FunctionSamples *ReadFooSamples = Reader->getSamplesFor(FooName);
    ASSERT_TRUE(ReadFooSamples != nullptr);
    if (Format != SampleProfileFormat::SPF_Compact_Binary) {
      ASSERT_EQ("_Z3fooi", ReadFooSamples->getName());
    }
    ASSERT_EQ(7711u, ReadFooSamples->getTotalSamples());
    ASSERT_EQ(610u, ReadFooSamples->getHeadSamples());

    FunctionSamples *ReadBarSamples = Reader->getSamplesFor(BarName);
    ASSERT_TRUE(ReadBarSamples != nullptr);
    if (Format != SampleProfileFormat::SPF_Compact_Binary) {
      ASSERT_EQ("_Z3bari", ReadBarSamples->getName());
    }
    ASSERT_EQ(20301u, ReadBarSamples->getTotalSamples());
    ASSERT_EQ(1437u, ReadBarSamples->getHeadSamples());
    ErrorOr<SampleRecord::CallTargetMap> CTMap =
        ReadBarSamples->findCallTargetMapAt(1, 0);
    ASSERT_FALSE(CTMap.getError());

    std::string MconstructGUID;
    StringRef MconstructRep =
        getRepInFormat(MconstructName, Format, MconstructGUID);
    std::string StringviewGUID;
    StringRef StringviewRep =
        getRepInFormat(StringviewName, Format, StringviewGUID);
    ASSERT_EQ(1000u, CTMap.get()[MconstructRep]);
    ASSERT_EQ(437u, CTMap.get()[StringviewRep]);

    auto VerifySummary = [](ProfileSummary &Summary) mutable {
      ASSERT_EQ(ProfileSummary::PSK_Sample, Summary.getKind());
      ASSERT_EQ(123603u, Summary.getTotalCount());
      ASSERT_EQ(6u, Summary.getNumCounts());
      ASSERT_EQ(2u, Summary.getNumFunctions());
      ASSERT_EQ(1437u, Summary.getMaxFunctionCount());
      ASSERT_EQ(60351u, Summary.getMaxCount());

      uint32_t Cutoff = 800000;
      auto Predicate = [&Cutoff](const ProfileSummaryEntry &PE) {
        return PE.Cutoff == Cutoff;
      };
      std::vector<ProfileSummaryEntry> &Details = Summary.getDetailedSummary();
      auto EightyPerc = find_if(Details, Predicate);
      Cutoff = 900000;
      auto NinetyPerc = find_if(Details, Predicate);
      Cutoff = 950000;
      auto NinetyFivePerc = find_if(Details, Predicate);
      Cutoff = 990000;
      auto NinetyNinePerc = find_if(Details, Predicate);
      ASSERT_EQ(60000u, EightyPerc->MinCount);
      ASSERT_EQ(60000u, NinetyPerc->MinCount);
      ASSERT_EQ(60000u, NinetyFivePerc->MinCount);
      ASSERT_EQ(610u, NinetyNinePerc->MinCount);
    };

    ProfileSummary &Summary = Reader->getSummary();
    VerifySummary(Summary);

    // Test that conversion of summary to and from Metadata works.
    Metadata *MD = Summary.getMD(Context);
    ASSERT_TRUE(MD);
    ProfileSummary *PS = ProfileSummary::getFromMD(MD);
    ASSERT_TRUE(PS);
    VerifySummary(*PS);
    delete PS;

    // Test that summary can be attached to and read back from module.
    M.setProfileSummary(MD, ProfileSummary::PSK_Sample);
    MD = M.getProfileSummary(/* IsCS */ false);
    ASSERT_TRUE(MD);
    PS = ProfileSummary::getFromMD(MD);
    ASSERT_TRUE(PS);
    VerifySummary(*PS);
    delete PS;
  }
};

TEST_F(SampleProfTest, roundtrip_text_profile) {
  testRoundTrip(SampleProfileFormat::SPF_Text, false);
}

TEST_F(SampleProfTest, roundtrip_raw_binary_profile) {
  testRoundTrip(SampleProfileFormat::SPF_Binary, false);
}

TEST_F(SampleProfTest, roundtrip_compact_binary_profile) {
  testRoundTrip(SampleProfileFormat::SPF_Compact_Binary, false);
}

TEST_F(SampleProfTest, remap_text_profile) {
  testRoundTrip(SampleProfileFormat::SPF_Text, true);
}

TEST_F(SampleProfTest, remap_raw_binary_profile) {
  testRoundTrip(SampleProfileFormat::SPF_Binary, true);
}

TEST_F(SampleProfTest, sample_overflow_saturation) {
  const uint64_t Max = std::numeric_limits<uint64_t>::max();
  sampleprof_error Result;

  StringRef FooName("_Z3fooi");
  FunctionSamples FooSamples;
  Result = FooSamples.addTotalSamples(1);
  ASSERT_EQ(Result, sampleprof_error::success);

  Result = FooSamples.addHeadSamples(1);
  ASSERT_EQ(Result, sampleprof_error::success);

  Result = FooSamples.addBodySamples(10, 0, 1);
  ASSERT_EQ(Result, sampleprof_error::success);

  Result = FooSamples.addTotalSamples(Max);
  ASSERT_EQ(Result, sampleprof_error::counter_overflow);
  ASSERT_EQ(FooSamples.getTotalSamples(), Max);

  Result = FooSamples.addHeadSamples(Max);
  ASSERT_EQ(Result, sampleprof_error::counter_overflow);
  ASSERT_EQ(FooSamples.getHeadSamples(), Max);

  Result = FooSamples.addBodySamples(10, 0, Max);
  ASSERT_EQ(Result, sampleprof_error::counter_overflow);
  ErrorOr<uint64_t> BodySamples = FooSamples.findSamplesAt(10, 0);
  ASSERT_FALSE(BodySamples.getError());
  ASSERT_EQ(BodySamples.get(), Max);
}

} // end anonymous namespace
