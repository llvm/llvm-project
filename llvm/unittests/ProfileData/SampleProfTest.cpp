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
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/ProfileData/SampleProfReader.h"
#include "llvm/ProfileData/SampleProfWriter.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"
#include <string>
#include <vector>

using namespace llvm;
using namespace sampleprof;

using llvm::unittest::TempFile;

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
        new raw_fd_ostream(Profile, EC, sys::fs::OF_None));
    auto WriterOrErr = SampleProfileWriter::create(OS, Format);
    ASSERT_TRUE(NoError(WriterOrErr.getError()));
    Writer = std::move(WriterOrErr.get());
  }

  void readProfile(const Module &M, StringRef Profile,
                   StringRef RemapFile = "") {
    auto FS = vfs::getRealFileSystem();
    auto ReaderOrErr = SampleProfileReader::create(
        std::string(Profile), Context, *FS, FSDiscriminatorPass::Base,
        std::string(RemapFile));
    ASSERT_TRUE(NoError(ReaderOrErr.getError()));
    Reader = std::move(ReaderOrErr.get());
    Reader->setModule(&M);
  }

  TempFile createRemapFile() {
    return TempFile("remapfile", "", R"(
      # Types 'int' and 'long' are equivalent
      type i l
      # Function names 'foo' and 'faux' are equivalent
      name 3foo 4faux
    )",
                    /*Unique*/ true);
  }

  // Verify profile summary is consistent in the roundtrip to and from
  // Metadata. \p AddPartialField is to choose whether the Metadata
  // contains the IsPartialProfile field which is optional.
  void verifyProfileSummary(ProfileSummary &Summary, Module &M,
                            const bool AddPartialField,
                            const bool AddPartialProfileRatioField) {
    LLVMContext &Context = M.getContext();
    const bool IsPartialProfile = Summary.isPartialProfile();
    const double PartialProfileRatio = Summary.getPartialProfileRatio();
    auto VerifySummary = [IsPartialProfile, PartialProfileRatio](
                             ProfileSummary &Summary) mutable {
      ASSERT_EQ(ProfileSummary::PSK_Sample, Summary.getKind());
      ASSERT_EQ(138211u, Summary.getTotalCount());
      ASSERT_EQ(10u, Summary.getNumCounts());
      ASSERT_EQ(4u, Summary.getNumFunctions());
      ASSERT_EQ(1437u, Summary.getMaxFunctionCount());
      ASSERT_EQ(60351u, Summary.getMaxCount());
      ASSERT_EQ(IsPartialProfile, Summary.isPartialProfile());
      ASSERT_EQ(PartialProfileRatio, Summary.getPartialProfileRatio());

      uint32_t Cutoff = 800000;
      auto Predicate = [&Cutoff](const ProfileSummaryEntry &PE) {
        return PE.Cutoff == Cutoff;
      };
      const std::vector<ProfileSummaryEntry> &Details =
          Summary.getDetailedSummary();
      auto EightyPerc = find_if(Details, Predicate);
      Cutoff = 900000;
      auto NinetyPerc = find_if(Details, Predicate);
      Cutoff = 950000;
      auto NinetyFivePerc = find_if(Details, Predicate);
      Cutoff = 990000;
      auto NinetyNinePerc = find_if(Details, Predicate);
      ASSERT_EQ(60000u, EightyPerc->MinCount);
      ASSERT_EQ(12557u, NinetyPerc->MinCount);
      ASSERT_EQ(12557u, NinetyFivePerc->MinCount);
      ASSERT_EQ(600u, NinetyNinePerc->MinCount);
    };
    VerifySummary(Summary);

    // Test that conversion of summary to and from Metadata works.
    Metadata *MD =
        Summary.getMD(Context, AddPartialField, AddPartialProfileRatioField);
    ASSERT_TRUE(MD);
    ProfileSummary *PS = ProfileSummary::getFromMD(MD);
    ASSERT_TRUE(PS);
    VerifySummary(*PS);
    delete PS;

    // Test that summary can be attached to and read back from module.
    M.eraseNamedMetadata(M.getOrInsertModuleFlagsMetadata());
    M.setProfileSummary(MD, ProfileSummary::PSK_Sample);
    MD = M.getProfileSummary(/* IsCS */ false);
    ASSERT_TRUE(MD);
    PS = ProfileSummary::getFromMD(MD);
    ASSERT_TRUE(PS);
    VerifySummary(*PS);
    delete PS;
  }

  void testRoundTrip(SampleProfileFormat Format, bool Remap, bool UseMD5) {
    TempFile ProfileFile("profile", "", "", /*Unique*/ true);
    createWriter(Format, ProfileFile.path());
    if (Format == SampleProfileFormat::SPF_Ext_Binary && UseMD5)
      static_cast<SampleProfileWriterExtBinary *>(Writer.get())->setUseMD5();

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

    // Add inline instance with name "_Z3gooi".
    StringRef GooName("_Z3gooi");
    auto &GooSamples =
        FooSamples.functionSamplesAt(LineLocation(7, 0))[GooName.str()];
    GooSamples.setName(GooName);
    GooSamples.addTotalSamples(502);
    GooSamples.addBodySamples(3, 0, 502);

    // Add inline instance with name "_Z3hooi".
    StringRef HooName("_Z3hooi");
    auto &HooSamples =
        GooSamples.functionSamplesAt(LineLocation(9, 0))[HooName.str()];
    HooSamples.setName(HooName);
    HooSamples.addTotalSamples(317);
    HooSamples.addBodySamples(4, 0, 317);

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

    StringRef BazName("_Z3bazi");
    FunctionSamples BazSamples;
    BazSamples.setName(BazName);
    BazSamples.addTotalSamples(12557);
    BazSamples.addHeadSamples(1257);
    BazSamples.addBodySamples(1, 0, 12557);

    StringRef BooName("_Z3booi");
    FunctionSamples BooSamples;
    BooSamples.setName(BooName);
    BooSamples.addTotalSamples(1232);
    BooSamples.addHeadSamples(1);
    BooSamples.addBodySamples(1, 0, 1232);

    SampleProfileMap Profiles;
    Profiles[FooName] = std::move(FooSamples);
    Profiles[BarName] = std::move(BarSamples);
    Profiles[BazName] = std::move(BazSamples);
    Profiles[BooName] = std::move(BooSamples);

    Module M("my_module", Context);
    FunctionType *fn_type =
        FunctionType::get(Type::getVoidTy(Context), {}, false);

    TempFile RemapFile(createRemapFile());
    if (Remap) {
      FooName = "_Z4fauxi";
      BarName = "_Z3barl";
      GooName = "_Z3gool";
      HooName = "_Z3hool";
    }

    M.getOrInsertFunction(FooName, fn_type);
    M.getOrInsertFunction(BarName, fn_type);
    M.getOrInsertFunction(BooName, fn_type);

    ProfileSymbolList List;
    if (Format == SampleProfileFormat::SPF_Ext_Binary) {
      List.add("zoo", true);
      List.add("moo", true);
    }
    Writer->setProfileSymbolList(&List);

    std::error_code EC;
    EC = Writer->write(Profiles);
    ASSERT_TRUE(NoError(EC));

    Writer->getOutputStream().flush();

    readProfile(M, ProfileFile.path(), RemapFile.path());
    EC = Reader->read();
    ASSERT_TRUE(NoError(EC));

    if (Format == SampleProfileFormat::SPF_Ext_Binary) {
      std::unique_ptr<ProfileSymbolList> ReaderList =
          Reader->getProfileSymbolList();
      ReaderList->contains("zoo");
      ReaderList->contains("moo");
    }

    FunctionSamples *ReadFooSamples = Reader->getSamplesFor(FooName);
    ASSERT_TRUE(ReadFooSamples != nullptr);
    if (!UseMD5) {
      ASSERT_EQ("_Z3fooi", ReadFooSamples->getName());
    }
    ASSERT_EQ(7711u, ReadFooSamples->getTotalSamples());
    ASSERT_EQ(610u, ReadFooSamples->getHeadSamples());

    // Try to find a FunctionSamples with GooName at given callsites containing
    // inline instance for GooName. Test the correct FunctionSamples can be
    // found with Remapper support.
    const FunctionSamples *ReadGooSamples =
        ReadFooSamples->findFunctionSamplesAt(LineLocation(7, 0), GooName,
                                              Reader->getRemapper());
    ASSERT_TRUE(ReadGooSamples != nullptr);
    ASSERT_EQ(502u, ReadGooSamples->getTotalSamples());

    // Try to find a FunctionSamples with GooName at given callsites containing
    // no inline instance for GooName. Test no FunctionSamples will be
    // found with Remapper support.
    const FunctionSamples *ReadGooSamplesAgain =
        ReadFooSamples->findFunctionSamplesAt(LineLocation(9, 0), GooName,
                                              Reader->getRemapper());
    ASSERT_TRUE(ReadGooSamplesAgain == nullptr);

    // The inline instance of Hoo is inside of the inline instance of Goo.
    // Try to find a FunctionSamples with HooName at given callsites containing
    // inline instance for HooName. Test the correct FunctionSamples can be
    // found with Remapper support.
    const FunctionSamples *ReadHooSamples =
        ReadGooSamples->findFunctionSamplesAt(LineLocation(9, 0), HooName,
                                              Reader->getRemapper());
    ASSERT_TRUE(ReadHooSamples != nullptr);
    ASSERT_EQ(317u, ReadHooSamples->getTotalSamples());

    FunctionSamples *ReadBarSamples = Reader->getSamplesFor(BarName);
    ASSERT_TRUE(ReadBarSamples != nullptr);
    if (!UseMD5) {
      ASSERT_EQ("_Z3bari", ReadBarSamples->getName());
    }
    ASSERT_EQ(20301u, ReadBarSamples->getTotalSamples());
    ASSERT_EQ(1437u, ReadBarSamples->getHeadSamples());
    ErrorOr<SampleRecord::CallTargetMap> CTMap =
        ReadBarSamples->findCallTargetMapAt(1, 0);
    ASSERT_FALSE(CTMap.getError());

    // Because _Z3bazi is not defined in module M, expect _Z3bazi's profile
    // is not loaded when the profile is ExtBinary format because this format
    // supports loading function profiles on demand.
    FunctionSamples *ReadBazSamples = Reader->getSamplesFor(BazName);
    if (Format == SampleProfileFormat::SPF_Ext_Binary) {
      ASSERT_TRUE(ReadBazSamples == nullptr);
      ASSERT_EQ(3u, Reader->getProfiles().size());
    } else {
      ASSERT_TRUE(ReadBazSamples != nullptr);
      ASSERT_EQ(12557u, ReadBazSamples->getTotalSamples());
      ASSERT_EQ(4u, Reader->getProfiles().size());
    }

    FunctionSamples *ReadBooSamples = Reader->getSamplesFor(BooName);
    ASSERT_TRUE(ReadBooSamples != nullptr);
    ASSERT_EQ(1232u, ReadBooSamples->getTotalSamples());

    std::string MconstructGUID;
    StringRef MconstructRep =
        getRepInFormat(MconstructName, UseMD5, MconstructGUID);
    std::string StringviewGUID;
    StringRef StringviewRep =
        getRepInFormat(StringviewName, UseMD5, StringviewGUID);
    ASSERT_EQ(1000u, CTMap.get()[MconstructRep]);
    ASSERT_EQ(437u, CTMap.get()[StringviewRep]);


    ProfileSummary &Summary = Reader->getSummary();
    Summary.setPartialProfile(true);
    verifyProfileSummary(Summary, M, true, false);

    Summary.setPartialProfile(false);
    verifyProfileSummary(Summary, M, true, false);

    verifyProfileSummary(Summary, M, false, false);

    Summary.setPartialProfile(true);
    Summary.setPartialProfileRatio(0.5);
    verifyProfileSummary(Summary, M, true, true);
  }

  void addFunctionSamples(SampleProfileMap *Smap, const char *Fname,
                          uint64_t TotalSamples, uint64_t HeadSamples) {
    StringRef Name(Fname);
    FunctionSamples FcnSamples;
    FcnSamples.setName(Name);
    FcnSamples.addTotalSamples(TotalSamples);
    FcnSamples.addHeadSamples(HeadSamples);
    FcnSamples.addBodySamples(1, 0, HeadSamples);
    (*Smap)[Name] = FcnSamples;
  }

  SampleProfileMap setupFcnSamplesForElisionTest(StringRef Policy) {
    SampleProfileMap Smap;
    addFunctionSamples(&Smap, "foo", uint64_t(20301), uint64_t(1437));
    if (Policy == "" || Policy == "all")
      return Smap;
    addFunctionSamples(&Smap, "foo.bar", uint64_t(20303), uint64_t(1439));
    if (Policy == "selected")
      return Smap;
    addFunctionSamples(&Smap, "foo.llvm.2465", uint64_t(20305), uint64_t(1441));
    return Smap;
  }

  void createFunctionWithSampleProfileElisionPolicy(Module *M,
                                                    const char *Fname,
                                                    StringRef Policy) {
    FunctionType *FnType =
        FunctionType::get(Type::getVoidTy(Context), {}, false);
    auto Inserted = M->getOrInsertFunction(Fname, FnType);
    auto Fcn = cast<Function>(Inserted.getCallee());
    if (Policy != "")
      Fcn->addFnAttr("sample-profile-suffix-elision-policy", Policy);
  }

  void setupModuleForElisionTest(Module *M, StringRef Policy) {
    createFunctionWithSampleProfileElisionPolicy(M, "foo", Policy);
    createFunctionWithSampleProfileElisionPolicy(M, "foo.bar", Policy);
    createFunctionWithSampleProfileElisionPolicy(M, "foo.llvm.2465", Policy);
  }

  void testSuffixElisionPolicy(SampleProfileFormat Format, StringRef Policy,
                               const StringMap<uint64_t> &Expected) {
    TempFile ProfileFile("profile", "", "", /*Unique*/ true);

    Module M("my_module", Context);
    setupModuleForElisionTest(&M, Policy);
    SampleProfileMap ProfMap = setupFcnSamplesForElisionTest(Policy);

    // write profile
    createWriter(Format, ProfileFile.path());
    std::error_code EC;
    EC = Writer->write(ProfMap);
    ASSERT_TRUE(NoError(EC));
    Writer->getOutputStream().flush();

    // read profile
    readProfile(M, ProfileFile.path());
    EC = Reader->read();
    ASSERT_TRUE(NoError(EC));

    for (auto I = Expected.begin(); I != Expected.end(); ++I) {
      uint64_t Esamples = uint64_t(-1);
      FunctionSamples *Samples = Reader->getSamplesFor(I->getKey());
      if (Samples != nullptr)
        Esamples = Samples->getTotalSamples();
      ASSERT_EQ(I->getValue(), Esamples);
    }
  }
};

TEST_F(SampleProfTest, roundtrip_text_profile) {
  testRoundTrip(SampleProfileFormat::SPF_Text, false, false);
}

TEST_F(SampleProfTest, roundtrip_raw_binary_profile) {
  testRoundTrip(SampleProfileFormat::SPF_Binary, false, false);
}

TEST_F(SampleProfTest, roundtrip_ext_binary_profile) {
  testRoundTrip(SampleProfileFormat::SPF_Ext_Binary, false, false);
}

TEST_F(SampleProfTest, roundtrip_md5_ext_binary_profile) {
  testRoundTrip(SampleProfileFormat::SPF_Ext_Binary, false, true);
}

TEST_F(SampleProfTest, remap_text_profile) {
  testRoundTrip(SampleProfileFormat::SPF_Text, true, false);
}

TEST_F(SampleProfTest, remap_raw_binary_profile) {
  testRoundTrip(SampleProfileFormat::SPF_Binary, true, false);
}

TEST_F(SampleProfTest, remap_ext_binary_profile) {
  testRoundTrip(SampleProfileFormat::SPF_Ext_Binary, true, false);
}

TEST_F(SampleProfTest, sample_overflow_saturation) {
  const uint64_t Max = std::numeric_limits<uint64_t>::max();
  sampleprof_error Result;

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

TEST_F(SampleProfTest, default_suffix_elision_text) {
  // Default suffix elision policy: strip everything after first dot.
  // This implies that all suffix variants will map to "foo", so
  // we don't expect to see any entries for them in the sample
  // profile.
  StringMap<uint64_t> Expected;
  Expected["foo"] = uint64_t(20301);
  Expected["foo.bar"] = uint64_t(-1);
  Expected["foo.llvm.2465"] = uint64_t(-1);
  testSuffixElisionPolicy(SampleProfileFormat::SPF_Text, "", Expected);
}

TEST_F(SampleProfTest, selected_suffix_elision_text) {
  // Profile is created and searched using the "selected"
  // suffix elision policy: we only strip a .XXX suffix if
  // it matches a pattern known to be generated by the compiler
  // (e.g. ".llvm.<digits>").
  StringMap<uint64_t> Expected;
  Expected["foo"] = uint64_t(20301);
  Expected["foo.bar"] = uint64_t(20303);
  Expected["foo.llvm.2465"] = uint64_t(-1);
  testSuffixElisionPolicy(SampleProfileFormat::SPF_Text, "selected", Expected);
}

TEST_F(SampleProfTest, none_suffix_elision_text) {
  // Profile is created and searched using the "none"
  // suffix elision policy: no stripping of suffixes at all.
  // Here we expect to see all variants in the profile.
  StringMap<uint64_t> Expected;
  Expected["foo"] = uint64_t(20301);
  Expected["foo.bar"] = uint64_t(20303);
  Expected["foo.llvm.2465"] = uint64_t(20305);
  testSuffixElisionPolicy(SampleProfileFormat::SPF_Text, "none", Expected);
}

} // end anonymous namespace
