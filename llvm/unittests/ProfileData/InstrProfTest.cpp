//===- unittest/ProfileData/InstrProfTest.cpp -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/ProfileData/IndexedMemProfData.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/ProfileData/InstrProfWriter.h"
#include "llvm/ProfileData/MemProf.h"
#include "llvm/ProfileData/MemProfData.inc"
#include "llvm/ProfileData/MemProfRadixTree.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <cstdarg>
#include <initializer_list>
#include <optional>

using namespace llvm;
using ::llvm::memprof::LineLocation;
using ::testing::ElementsAre;
using ::testing::EndsWith;
using ::testing::IsSubsetOf;
using ::testing::Pair;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

[[nodiscard]] static ::testing::AssertionResult
ErrorEquals(instrprof_error Expected, Error E) {
  instrprof_error Found;
  std::string FoundMsg;
  handleAllErrors(std::move(E), [&](const InstrProfError &IPE) {
    Found = IPE.get();
    FoundMsg = IPE.message();
  });
  if (Expected == Found)
    return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << "error: " << FoundMsg << "\n";
}

namespace llvm {
bool operator==(const TemporalProfTraceTy &lhs,
                const TemporalProfTraceTy &rhs) {
  return lhs.Weight == rhs.Weight &&
         lhs.FunctionNameRefs == rhs.FunctionNameRefs;
}
} // end namespace llvm

namespace {

struct InstrProfTest : ::testing::Test {
  InstrProfWriter Writer;
  std::unique_ptr<IndexedInstrProfReader> Reader;

  void SetUp() override { Writer.setOutputSparse(false); }

  void readProfile(std::unique_ptr<MemoryBuffer> Profile,
                   std::unique_ptr<MemoryBuffer> Remapping = nullptr) {
    auto ReaderOrErr = IndexedInstrProfReader::create(std::move(Profile),
                                                      std::move(Remapping));
    EXPECT_THAT_ERROR(ReaderOrErr.takeError(), Succeeded());
    Reader = std::move(ReaderOrErr.get());
  }
};

struct SparseInstrProfTest : public InstrProfTest {
  void SetUp() override { Writer.setOutputSparse(true); }
};

struct InstrProfReaderWriterTest
    : public InstrProfTest,
      public ::testing::WithParamInterface<
          std::tuple<bool, uint64_t, llvm::endianness>> {
  void SetUp() override { Writer.setOutputSparse(std::get<0>(GetParam())); }
  void TearDown() override {
    // Reset writer value profile data endianness after each test case. Note
    // it's not necessary to reset reader value profile endianness for each test
    // case. Each test case creates a new reader; at reader initialization time,
    // it uses the endianness from hash table object (which is little by
    // default).
    Writer.setValueProfDataEndianness(llvm::endianness::little);
  }

  uint64_t getProfWeight() const { return std::get<1>(GetParam()); }

  llvm::endianness getEndianness() const { return std::get<2>(GetParam()); }
};

struct MaybeSparseInstrProfTest : public InstrProfTest,
                                  public ::testing::WithParamInterface<bool> {
  void SetUp() override { Writer.setOutputSparse(GetParam()); }
};

TEST_P(MaybeSparseInstrProfTest, write_and_read_empty_profile) {
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));
  ASSERT_TRUE(Reader->begin() == Reader->end());
}

static const auto Err = [](Error E) {
  consumeError(std::move(E));
  FAIL();
};

TEST_P(MaybeSparseInstrProfTest, write_and_read_one_function) {
  Writer.addRecord({"foo", 0x1234, {1, 2, 3, 4}}, Err);
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  auto I = Reader->begin(), E = Reader->end();
  ASSERT_TRUE(I != E);
  ASSERT_EQ(StringRef("foo"), I->Name);
  ASSERT_EQ(0x1234U, I->Hash);
  ASSERT_EQ(4U, I->Counts.size());
  ASSERT_EQ(1U, I->Counts[0]);
  ASSERT_EQ(2U, I->Counts[1]);
  ASSERT_EQ(3U, I->Counts[2]);
  ASSERT_EQ(4U, I->Counts[3]);
  ASSERT_TRUE(++I == E);
}

TEST_P(MaybeSparseInstrProfTest, get_instr_prof_record) {
  Writer.addRecord({"foo", 0x1234, {1, 2}}, Err);
  Writer.addRecord({"foo", 0x1235, {3, 4}}, Err);
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  auto R = Reader->getInstrProfRecord("foo", 0x1234);
  EXPECT_THAT_ERROR(R.takeError(), Succeeded());
  ASSERT_EQ(2U, R->Counts.size());
  ASSERT_EQ(1U, R->Counts[0]);
  ASSERT_EQ(2U, R->Counts[1]);

  R = Reader->getInstrProfRecord("foo", 0x1235);
  EXPECT_THAT_ERROR(R.takeError(), Succeeded());
  ASSERT_EQ(2U, R->Counts.size());
  ASSERT_EQ(3U, R->Counts[0]);
  ASSERT_EQ(4U, R->Counts[1]);

  R = Reader->getInstrProfRecord("foo", 0x5678);
  ASSERT_TRUE(ErrorEquals(instrprof_error::hash_mismatch, R.takeError()));

  R = Reader->getInstrProfRecord("bar", 0x1234);
  ASSERT_TRUE(ErrorEquals(instrprof_error::unknown_function, R.takeError()));
}

TEST_P(MaybeSparseInstrProfTest, get_function_counts) {
  Writer.addRecord({"foo", 0x1234, {1, 2}}, Err);
  Writer.addRecord({"foo", 0x1235, {3, 4}}, Err);
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  std::vector<uint64_t> Counts;
  EXPECT_THAT_ERROR(Reader->getFunctionCounts("foo", 0x1234, Counts),
                    Succeeded());
  ASSERT_EQ(2U, Counts.size());
  ASSERT_EQ(1U, Counts[0]);
  ASSERT_EQ(2U, Counts[1]);

  EXPECT_THAT_ERROR(Reader->getFunctionCounts("foo", 0x1235, Counts),
                    Succeeded());
  ASSERT_EQ(2U, Counts.size());
  ASSERT_EQ(3U, Counts[0]);
  ASSERT_EQ(4U, Counts[1]);

  Error E1 = Reader->getFunctionCounts("foo", 0x5678, Counts);
  ASSERT_TRUE(ErrorEquals(instrprof_error::hash_mismatch, std::move(E1)));

  Error E2 = Reader->getFunctionCounts("bar", 0x1234, Counts);
  ASSERT_TRUE(ErrorEquals(instrprof_error::unknown_function, std::move(E2)));
}

// Profile data is copied from general.proftext
TEST_F(InstrProfTest, get_profile_summary) {
  Writer.addRecord({"func1", 0x1234, {97531}}, Err);
  Writer.addRecord({"func2", 0x1234, {0, 0}}, Err);
  Writer.addRecord(
      {"func3",
       0x1234,
       {2305843009213693952, 1152921504606846976, 576460752303423488,
        288230376151711744, 144115188075855872, 72057594037927936}},
      Err);
  Writer.addRecord({"func4", 0x1234, {0}}, Err);
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  auto VerifySummary = [](ProfileSummary &IPS) mutable {
    ASSERT_EQ(ProfileSummary::PSK_Instr, IPS.getKind());
    ASSERT_EQ(2305843009213693952U, IPS.getMaxFunctionCount());
    ASSERT_EQ(2305843009213693952U, IPS.getMaxCount());
    ASSERT_EQ(10U, IPS.getNumCounts());
    ASSERT_EQ(4539628424389557499U, IPS.getTotalCount());
    const std::vector<ProfileSummaryEntry> &Details = IPS.getDetailedSummary();
    uint32_t Cutoff = 800000;
    auto Predicate = [&Cutoff](const ProfileSummaryEntry &PE) {
      return PE.Cutoff == Cutoff;
    };
    auto EightyPerc = find_if(Details, Predicate);
    Cutoff = 900000;
    auto NinetyPerc = find_if(Details, Predicate);
    Cutoff = 950000;
    auto NinetyFivePerc = find_if(Details, Predicate);
    Cutoff = 990000;
    auto NinetyNinePerc = find_if(Details, Predicate);
    ASSERT_EQ(576460752303423488U, EightyPerc->MinCount);
    ASSERT_EQ(288230376151711744U, NinetyPerc->MinCount);
    ASSERT_EQ(288230376151711744U, NinetyFivePerc->MinCount);
    ASSERT_EQ(72057594037927936U, NinetyNinePerc->MinCount);
  };
  ProfileSummary &PS = Reader->getSummary(/* IsCS */ false);
  VerifySummary(PS);

  // Test that conversion of summary to and from Metadata works.
  LLVMContext Context;
  Metadata *MD = PS.getMD(Context);
  ASSERT_TRUE(MD);
  ProfileSummary *PSFromMD = ProfileSummary::getFromMD(MD);
  ASSERT_TRUE(PSFromMD);
  VerifySummary(*PSFromMD);
  delete PSFromMD;

  // Test that summary can be attached to and read back from module.
  Module M("my_module", Context);
  M.setProfileSummary(MD, ProfileSummary::PSK_Instr);
  MD = M.getProfileSummary(/* IsCS */ false);
  ASSERT_TRUE(MD);
  PSFromMD = ProfileSummary::getFromMD(MD);
  ASSERT_TRUE(PSFromMD);
  VerifySummary(*PSFromMD);
  delete PSFromMD;
}

TEST_F(InstrProfTest, test_writer_merge) {
  Writer.addRecord({"func1", 0x1234, {42}}, Err);

  InstrProfWriter Writer2;
  Writer2.addRecord({"func2", 0x1234, {0, 0}}, Err);

  Writer.mergeRecordsFromWriter(std::move(Writer2), Err);

  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  auto R = Reader->getInstrProfRecord("func1", 0x1234);
  EXPECT_THAT_ERROR(R.takeError(), Succeeded());
  ASSERT_EQ(1U, R->Counts.size());
  ASSERT_EQ(42U, R->Counts[0]);

  R = Reader->getInstrProfRecord("func2", 0x1234);
  EXPECT_THAT_ERROR(R.takeError(), Succeeded());
  ASSERT_EQ(2U, R->Counts.size());
  ASSERT_EQ(0U, R->Counts[0]);
  ASSERT_EQ(0U, R->Counts[1]);
}

TEST_F(InstrProfTest, test_merge_temporal_prof_traces_truncated) {
  uint64_t ReservoirSize = 10;
  uint64_t MaxTraceLength = 2;
  InstrProfWriter Writer(/*Sparse=*/false, ReservoirSize, MaxTraceLength);
  ASSERT_THAT_ERROR(Writer.mergeProfileKind(InstrProfKind::TemporalProfile),
                    Succeeded());

  TemporalProfTraceTy LargeTrace, SmallTrace;
  LargeTrace.FunctionNameRefs = {IndexedInstrProf::ComputeHash("foo"),
                                 IndexedInstrProf::ComputeHash("bar"),
                                 IndexedInstrProf::ComputeHash("goo")};
  SmallTrace.FunctionNameRefs = {IndexedInstrProf::ComputeHash("foo"),
                                 IndexedInstrProf::ComputeHash("bar")};

  SmallVector<TemporalProfTraceTy, 4> Traces = {LargeTrace, SmallTrace};
  Writer.addTemporalProfileTraces(Traces, 2);

  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  ASSERT_TRUE(Reader->hasTemporalProfile());
  EXPECT_EQ(Reader->getTemporalProfTraceStreamSize(), 2U);
  EXPECT_THAT(Reader->getTemporalProfTraces(),
              UnorderedElementsAre(SmallTrace, SmallTrace));
}

TEST_F(InstrProfTest, test_merge_traces_from_writer) {
  uint64_t ReservoirSize = 10;
  uint64_t MaxTraceLength = 10;
  InstrProfWriter Writer(/*Sparse=*/false, ReservoirSize, MaxTraceLength);
  InstrProfWriter Writer2(/*Sparse=*/false, ReservoirSize, MaxTraceLength);
  ASSERT_THAT_ERROR(Writer.mergeProfileKind(InstrProfKind::TemporalProfile),
                    Succeeded());
  ASSERT_THAT_ERROR(Writer2.mergeProfileKind(InstrProfKind::TemporalProfile),
                    Succeeded());

  TemporalProfTraceTy FooTrace, BarTrace;
  FooTrace.FunctionNameRefs = {IndexedInstrProf::ComputeHash("foo")};
  BarTrace.FunctionNameRefs = {IndexedInstrProf::ComputeHash("bar")};

  SmallVector<TemporalProfTraceTy, 4> Traces1({FooTrace}), Traces2({BarTrace});
  Writer.addTemporalProfileTraces(Traces1, 1);
  Writer2.addTemporalProfileTraces(Traces2, 1);
  Writer.mergeRecordsFromWriter(std::move(Writer2), Err);

  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  ASSERT_TRUE(Reader->hasTemporalProfile());
  EXPECT_EQ(Reader->getTemporalProfTraceStreamSize(), 2U);
  EXPECT_THAT(Reader->getTemporalProfTraces(),
              UnorderedElementsAre(FooTrace, BarTrace));
}

TEST_F(InstrProfTest, test_merge_traces_sampled) {
  uint64_t ReservoirSize = 3;
  uint64_t MaxTraceLength = 10;
  InstrProfWriter Writer(/*Sparse=*/false, ReservoirSize, MaxTraceLength);
  ASSERT_THAT_ERROR(Writer.mergeProfileKind(InstrProfKind::TemporalProfile),
                    Succeeded());

  TemporalProfTraceTy FooTrace, BarTrace, GooTrace;
  FooTrace.FunctionNameRefs = {IndexedInstrProf::ComputeHash("foo")};
  BarTrace.FunctionNameRefs = {IndexedInstrProf::ComputeHash("bar")};
  GooTrace.FunctionNameRefs = {IndexedInstrProf::ComputeHash("Goo")};

  // Add some sampled traces
  SmallVector<TemporalProfTraceTy, 4> SampledTraces = {FooTrace, BarTrace,
                                                       GooTrace};
  Writer.addTemporalProfileTraces(SampledTraces, 5);
  // Add some unsampled traces
  SmallVector<TemporalProfTraceTy, 4> UnsampledTraces = {BarTrace, GooTrace};
  Writer.addTemporalProfileTraces(UnsampledTraces, 2);
  UnsampledTraces = {FooTrace};
  Writer.addTemporalProfileTraces(UnsampledTraces, 1);

  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  ASSERT_TRUE(Reader->hasTemporalProfile());
  EXPECT_EQ(Reader->getTemporalProfTraceStreamSize(), 8U);
  // Check that we have a subset of all the traces we added
  EXPECT_THAT(Reader->getTemporalProfTraces(), SizeIs(ReservoirSize));
  EXPECT_THAT(
      Reader->getTemporalProfTraces(),
      IsSubsetOf({FooTrace, BarTrace, GooTrace, BarTrace, GooTrace, FooTrace}));
}

using ::llvm::memprof::IndexedMemProfData;
using ::llvm::memprof::IndexedMemProfRecord;
using ::llvm::memprof::MemInfoBlock;

IndexedMemProfData getMemProfDataForTest() {
  IndexedMemProfData MemProfData;

  MemProfData.Frames.insert({0, {0x123, 1, 2, false}});
  MemProfData.Frames.insert({1, {0x345, 3, 4, true}});
  MemProfData.Frames.insert({2, {0x125, 5, 6, false}});
  MemProfData.Frames.insert({3, {0x567, 7, 8, true}});
  MemProfData.Frames.insert({4, {0x124, 5, 6, false}});
  MemProfData.Frames.insert({5, {0x789, 8, 9, true}});

  MemProfData.CallStacks.insert({0x111, {0, 1}});
  MemProfData.CallStacks.insert({0x222, {2, 3}});
  MemProfData.CallStacks.insert({0x333, {4, 5}});

  return MemProfData;
}

// Populate all of the fields of MIB.
MemInfoBlock makeFullMIB() {
  MemInfoBlock MIB;
#define MIBEntryDef(NameTag, Name, Type) MIB.NameTag;
#include "llvm/ProfileData/MIBEntryDef.inc"
#undef MIBEntryDef
  return MIB;
}

// Populate those fields returned by getHotColdSchema.
MemInfoBlock makePartialMIB() {
  MemInfoBlock MIB;
  MIB.AllocCount = 1;
  MIB.TotalSize = 5;
  MIB.TotalLifetime = 10;
  MIB.TotalLifetimeAccessDensity = 23;
  return MIB;
}

IndexedMemProfRecord
makeRecord(std::initializer_list<::llvm::memprof::CallStackId> AllocFrames,
           std::initializer_list<::llvm::memprof::CallStackId> CallSiteFrames,
           const MemInfoBlock &Block, const memprof::MemProfSchema &Schema) {
  IndexedMemProfRecord MR;
  for (const auto &CSId : AllocFrames)
    MR.AllocSites.emplace_back(CSId, Block, Schema);
  for (const auto &CSId : CallSiteFrames)
    MR.CallSites.push_back(llvm::memprof::IndexedCallSiteInfo(CSId));
  return MR;
}

MATCHER_P(EqualsRecord, Want, "") {
  const memprof::MemProfRecord &Got = arg;

  auto PrintAndFail = [&]() {
    std::string Buffer;
    llvm::raw_string_ostream OS(Buffer);
    OS << "Want:\n";
    Want.print(OS);
    OS << "Got:\n";
    Got.print(OS);
    *result_listener << "MemProf Record differs!\n" << Buffer;
    return false;
  };

  if (Want.AllocSites.size() != Got.AllocSites.size())
    return PrintAndFail();
  if (Want.CallSites.size() != Got.CallSites.size())
    return PrintAndFail();

  for (size_t I = 0; I < Got.AllocSites.size(); I++) {
    if (Want.AllocSites[I].Info != Got.AllocSites[I].Info)
      return PrintAndFail();
    if (Want.AllocSites[I].CallStack != Got.AllocSites[I].CallStack)
      return PrintAndFail();
  }

  for (size_t I = 0; I < Got.CallSites.size(); I++) {
    if (Want.CallSites[I] != Got.CallSites[I])
      return PrintAndFail();
  }
  return true;
}

TEST_F(InstrProfTest, test_memprof_v4_full_schema) {
  const MemInfoBlock MIB = makeFullMIB();

  Writer.setMemProfVersionRequested(memprof::Version4);
  Writer.setMemProfFullSchema(true);

  ASSERT_THAT_ERROR(Writer.mergeProfileKind(InstrProfKind::MemProf),
                    Succeeded());

  const IndexedMemProfRecord IndexedMR = makeRecord(
      /*AllocFrames=*/{0x111, 0x222},
      /*CallSiteFrames=*/{0x333}, MIB, memprof::getFullSchema());
  IndexedMemProfData MemProfData = getMemProfDataForTest();
  MemProfData.Records.try_emplace(0x9999, IndexedMR);
  Writer.addMemProfData(MemProfData, Err);

  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  auto RecordOr = Reader->getMemProfRecord(0x9999);
  ASSERT_THAT_ERROR(RecordOr.takeError(), Succeeded());
  const memprof::MemProfRecord &Record = RecordOr.get();

  memprof::IndexedCallstackIdConverter CSIdConv(MemProfData);

  const ::llvm::memprof::MemProfRecord WantRecord =
      IndexedMR.toMemProfRecord(CSIdConv);
  ASSERT_EQ(CSIdConv.FrameIdConv.LastUnmappedId, std::nullopt)
      << "could not map frame id: " << *CSIdConv.FrameIdConv.LastUnmappedId;
  ASSERT_EQ(CSIdConv.CSIdConv.LastUnmappedId, std::nullopt)
      << "could not map call stack id: " << *CSIdConv.CSIdConv.LastUnmappedId;
  EXPECT_THAT(WantRecord, EqualsRecord(Record));
}

TEST_F(InstrProfTest, test_memprof_v4_partial_schema) {
  const MemInfoBlock MIB = makePartialMIB();

  Writer.setMemProfVersionRequested(memprof::Version4);
  Writer.setMemProfFullSchema(false);

  ASSERT_THAT_ERROR(Writer.mergeProfileKind(InstrProfKind::MemProf),
                    Succeeded());

  const IndexedMemProfRecord IndexedMR = makeRecord(
      /*AllocFrames=*/{0x111, 0x222},
      /*CallSiteFrames=*/{0x333}, MIB, memprof::getHotColdSchema());
  IndexedMemProfData MemProfData = getMemProfDataForTest();
  MemProfData.Records.try_emplace(0x9999, IndexedMR);
  Writer.addMemProfData(MemProfData, Err);

  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  auto RecordOr = Reader->getMemProfRecord(0x9999);
  ASSERT_THAT_ERROR(RecordOr.takeError(), Succeeded());
  const memprof::MemProfRecord &Record = RecordOr.get();

  memprof::IndexedCallstackIdConverter CSIdConv(MemProfData);

  const ::llvm::memprof::MemProfRecord WantRecord =
      IndexedMR.toMemProfRecord(CSIdConv);
  ASSERT_EQ(CSIdConv.FrameIdConv.LastUnmappedId, std::nullopt)
      << "could not map frame id: " << *CSIdConv.FrameIdConv.LastUnmappedId;
  ASSERT_EQ(CSIdConv.CSIdConv.LastUnmappedId, std::nullopt)
      << "could not map call stack id: " << *CSIdConv.CSIdConv.LastUnmappedId;
  EXPECT_THAT(WantRecord, EqualsRecord(Record));
}

TEST_F(InstrProfTest, test_caller_callee_pairs) {
  const MemInfoBlock MIB = makePartialMIB();

  Writer.setMemProfVersionRequested(memprof::Version3);
  Writer.setMemProfFullSchema(false);

  ASSERT_THAT_ERROR(Writer.mergeProfileKind(InstrProfKind::MemProf),
                    Succeeded());

  // Call Hierarchy
  //
  // Function GUID:0x123
  //   Line: 1, Column: 2
  //     Function GUID: 0x234
  //       Line: 3, Column: 4
  //         new(...)
  //   Line: 5, Column: 6
  //     Function GUID: 0x345
  //       Line: 7, Column: 8
  //         new(...)

  const IndexedMemProfRecord IndexedMR = makeRecord(
      /*AllocFrames=*/{0x111, 0x222},
      /*CallSiteFrames=*/{}, MIB, memprof::getHotColdSchema());

  IndexedMemProfData MemProfData;
  MemProfData.Frames.try_emplace(0, 0x123, 1, 2, false);
  MemProfData.Frames.try_emplace(1, 0x234, 3, 4, true);
  MemProfData.Frames.try_emplace(2, 0x123, 5, 6, false);
  MemProfData.Frames.try_emplace(3, 0x345, 7, 8, true);
  MemProfData.CallStacks.try_emplace(
      0x111, std::initializer_list<memprof::FrameId>{1, 0});
  MemProfData.CallStacks.try_emplace(
      0x222, std::initializer_list<memprof::FrameId>{3, 2});
  MemProfData.Records.try_emplace(0x9999, IndexedMR);
  Writer.addMemProfData(MemProfData, Err);

  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  auto Pairs = Reader->getMemProfCallerCalleePairs();
  ASSERT_THAT(Pairs, SizeIs(3));

  auto It = Pairs.find(0x123);
  ASSERT_NE(It, Pairs.end());
  EXPECT_THAT(It->second, ElementsAre(Pair(LineLocation(1, 2), 0x234U),
                                      Pair(LineLocation(5, 6), 0x345U)));

  It = Pairs.find(0x234);
  ASSERT_NE(It, Pairs.end());
  EXPECT_THAT(It->second, ElementsAre(Pair(LineLocation(3, 4), 0U)));

  It = Pairs.find(0x345);
  ASSERT_NE(It, Pairs.end());
  EXPECT_THAT(It->second, ElementsAre(Pair(LineLocation(7, 8), 0U)));
}

TEST_F(InstrProfTest, test_memprof_getrecord_error) {
  ASSERT_THAT_ERROR(Writer.mergeProfileKind(InstrProfKind::MemProf),
                    Succeeded());

  Writer.setMemProfVersionRequested(memprof::Version3);
  // Generate an empty profile.
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  // Missing functions give a unknown_function error.
  auto RecordOr = Reader->getMemProfRecord(0x1111);
  ASSERT_TRUE(
      ErrorEquals(instrprof_error::unknown_function, RecordOr.takeError()));
}

TEST_F(InstrProfTest, test_memprof_merge) {
  Writer.addRecord({"func1", 0x1234, {42}}, Err);

  InstrProfWriter Writer2;
  Writer2.setMemProfVersionRequested(memprof::Version3);
  ASSERT_THAT_ERROR(Writer2.mergeProfileKind(InstrProfKind::MemProf),
                    Succeeded());

  const IndexedMemProfRecord IndexedMR = makeRecord(
      /*AllocFrames=*/{0x111, 0x222},
      /*CallSiteFrames=*/{}, makePartialMIB(), memprof::getHotColdSchema());

  IndexedMemProfData MemProfData = getMemProfDataForTest();
  MemProfData.Records.try_emplace(0x9999, IndexedMR);
  Writer2.addMemProfData(MemProfData, Err);

  ASSERT_THAT_ERROR(Writer.mergeProfileKind(Writer2.getProfileKind()),
                    Succeeded());
  Writer.mergeRecordsFromWriter(std::move(Writer2), Err);

  Writer.setMemProfVersionRequested(memprof::Version3);
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  auto R = Reader->getInstrProfRecord("func1", 0x1234);
  EXPECT_THAT_ERROR(R.takeError(), Succeeded());
  ASSERT_EQ(1U, R->Counts.size());
  ASSERT_EQ(42U, R->Counts[0]);

  auto RecordOr = Reader->getMemProfRecord(0x9999);
  ASSERT_THAT_ERROR(RecordOr.takeError(), Succeeded());
  const memprof::MemProfRecord &Record = RecordOr.get();

  std::optional<memprof::FrameId> LastUnmappedFrameId;

  memprof::IndexedCallstackIdConverter CSIdConv(MemProfData);

  const ::llvm::memprof::MemProfRecord WantRecord =
      IndexedMR.toMemProfRecord(CSIdConv);
  ASSERT_EQ(LastUnmappedFrameId, std::nullopt)
      << "could not map frame id: " << *LastUnmappedFrameId;
  EXPECT_THAT(WantRecord, EqualsRecord(Record));
}

TEST_F(InstrProfTest, test_irpgo_function_name) {
  LLVMContext Ctx;
  auto M = std::make_unique<Module>("MyModule.cpp", Ctx);
  auto *FTy = FunctionType::get(Type::getVoidTy(Ctx), /*isVarArg=*/false);

  std::vector<std::tuple<StringRef, Function::LinkageTypes, StringRef>> Data;
  Data.emplace_back("ExternalFoo", Function::ExternalLinkage, "ExternalFoo");
  Data.emplace_back("InternalFoo", Function::InternalLinkage,
                    "MyModule.cpp;InternalFoo");
  Data.emplace_back("\01-[C dynamicFoo:]", Function::ExternalLinkage,
                    "-[C dynamicFoo:]");
  Data.emplace_back("\01-[C internalFoo:]", Function::InternalLinkage,
                    "MyModule.cpp;-[C internalFoo:]");

  for (auto &[Name, Linkage, ExpectedIRPGOFuncName] : Data)
    Function::Create(FTy, Linkage, Name, M.get());

  for (auto &[Name, Linkage, ExpectedIRPGOFuncName] : Data) {
    auto *F = M->getFunction(Name);
    auto IRPGOFuncName = getIRPGOFuncName(*F);
    EXPECT_EQ(IRPGOFuncName, ExpectedIRPGOFuncName);

    auto [Filename, ParsedIRPGOFuncName] = getParsedIRPGOName(IRPGOFuncName);
    StringRef ExpectedParsedIRPGOFuncName = IRPGOFuncName;
    if (ExpectedParsedIRPGOFuncName.consume_front("MyModule.cpp;")) {
      EXPECT_EQ(Filename, "MyModule.cpp");
    } else {
      EXPECT_EQ(Filename, "");
    }
    EXPECT_EQ(ParsedIRPGOFuncName, ExpectedParsedIRPGOFuncName);
  }
}

TEST_F(InstrProfTest, test_pgo_function_name) {
  LLVMContext Ctx;
  auto M = std::make_unique<Module>("MyModule.cpp", Ctx);
  auto *FTy = FunctionType::get(Type::getVoidTy(Ctx), /*isVarArg=*/false);

  std::vector<std::tuple<StringRef, Function::LinkageTypes, StringRef>> Data;
  Data.emplace_back("ExternalFoo", Function::ExternalLinkage, "ExternalFoo");
  Data.emplace_back("InternalFoo", Function::InternalLinkage,
                    "MyModule.cpp:InternalFoo");
  Data.emplace_back("\01-[C externalFoo:]", Function::ExternalLinkage,
                    "-[C externalFoo:]");
  Data.emplace_back("\01-[C internalFoo:]", Function::InternalLinkage,
                    "MyModule.cpp:-[C internalFoo:]");

  for (auto &[Name, Linkage, ExpectedPGOFuncName] : Data)
    Function::Create(FTy, Linkage, Name, M.get());

  for (auto &[Name, Linkage, ExpectedPGOFuncName] : Data) {
    auto *F = M->getFunction(Name);
    EXPECT_EQ(getPGOFuncName(*F), ExpectedPGOFuncName);
  }
}

TEST_F(InstrProfTest, test_irpgo_read_deprecated_names) {
  LLVMContext Ctx;
  auto M = std::make_unique<Module>("MyModule.cpp", Ctx);
  auto *FTy = FunctionType::get(Type::getVoidTy(Ctx), /*isVarArg=*/false);
  auto *InternalFooF =
      Function::Create(FTy, Function::InternalLinkage, "InternalFoo", M.get());
  auto *ExternalFooF =
      Function::Create(FTy, Function::ExternalLinkage, "ExternalFoo", M.get());

  auto *InternalBarF =
      Function::Create(FTy, Function::InternalLinkage, "InternalBar", M.get());
  auto *ExternalBarF =
      Function::Create(FTy, Function::ExternalLinkage, "ExternalBar", M.get());

  Writer.addRecord({getIRPGOFuncName(*InternalFooF), 0x1234, {1}}, Err);
  Writer.addRecord({getIRPGOFuncName(*ExternalFooF), 0x5678, {1}}, Err);
  // Write a record with a deprecated name
  Writer.addRecord({getPGOFuncName(*InternalBarF), 0x1111, {2}}, Err);
  Writer.addRecord({getPGOFuncName(*ExternalBarF), 0x2222, {2}}, Err);

  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  EXPECT_THAT_EXPECTED(
      Reader->getInstrProfRecord(getIRPGOFuncName(*InternalFooF), 0x1234,
                                 getPGOFuncName(*InternalFooF)),
      Succeeded());
  EXPECT_THAT_EXPECTED(
      Reader->getInstrProfRecord(getIRPGOFuncName(*ExternalFooF), 0x5678,
                                 getPGOFuncName(*ExternalFooF)),
      Succeeded());
  // Ensure we can still read this old record name
  EXPECT_THAT_EXPECTED(
      Reader->getInstrProfRecord(getIRPGOFuncName(*InternalBarF), 0x1111,
                                 getPGOFuncName(*InternalBarF)),
      Succeeded());
  EXPECT_THAT_EXPECTED(
      Reader->getInstrProfRecord(getIRPGOFuncName(*ExternalBarF), 0x2222,
                                 getPGOFuncName(*ExternalBarF)),
      Succeeded());
}

// callee1 to callee6 are from vtable1 to vtable6 respectively.
static const char callee1[] = "callee1";
static const char callee2[] = "callee2";
static const char callee3[] = "callee3";
static const char callee4[] = "callee4";
static const char callee5[] = "callee5";
static const char callee6[] = "callee6";
// callee7 and callee8 are not from any vtables.
static const char callee7[] = "callee7";
static const char callee8[] = "callee8";
// 'callee' is primarily used to create multiple-element vtables.
static const char callee[] = "callee";
static const uint64_t vtable1[] = {uint64_t(callee), uint64_t(callee1)};
static const uint64_t vtable2[] = {uint64_t(callee2), uint64_t(callee)};
static const uint64_t vtable3[] = {
    uint64_t(callee),
    uint64_t(callee3),
};
static const uint64_t vtable4[] = {uint64_t(callee4), uint64_t(callee)};
static const uint64_t vtable5[] = {uint64_t(callee5), uint64_t(callee)};
static const uint64_t vtable6[] = {uint64_t(callee6), uint64_t(callee)};

// Returns the address of callee with a numbered suffix in vtable.
static uint64_t getCalleeAddress(const uint64_t *vtableAddr) {
  uint64_t CalleeAddr;
  // Callee with a numbered suffix is the 2nd element in vtable1 and vtable3,
  // and the 1st element in the rest of vtables.
  if (vtableAddr == vtable1 || vtableAddr == vtable3)
    CalleeAddr = uint64_t(vtableAddr) + 8;
  else
    CalleeAddr = uint64_t(vtableAddr);
  return CalleeAddr;
}

TEST_P(InstrProfReaderWriterTest, icall_and_vtable_data_read_write) {
  NamedInstrProfRecord Record1("caller", 0x1234, {1, 2});

  // 4 indirect call value sites.
  {
    Record1.reserveSites(IPVK_IndirectCallTarget, 4);
    InstrProfValueData VD0[] = {
        {(uint64_t)callee1, 1}, {(uint64_t)callee2, 2}, {(uint64_t)callee3, 3}};
    Record1.addValueData(IPVK_IndirectCallTarget, 0, VD0, nullptr);
    // No value profile data at the second site.
    Record1.addValueData(IPVK_IndirectCallTarget, 1, {}, nullptr);
    InstrProfValueData VD2[] = {{(uint64_t)callee1, 1}, {(uint64_t)callee2, 2}};
    Record1.addValueData(IPVK_IndirectCallTarget, 2, VD2, nullptr);
    InstrProfValueData VD3[] = {{(uint64_t)callee7, 1}, {(uint64_t)callee8, 2}};
    Record1.addValueData(IPVK_IndirectCallTarget, 3, VD3, nullptr);
  }

  // 2 vtable value sites.
  {
    InstrProfValueData VD0[] = {
        {getCalleeAddress(vtable1), 1},
        {getCalleeAddress(vtable2), 2},
        {getCalleeAddress(vtable3), 3},
    };
    InstrProfValueData VD2[] = {
        {getCalleeAddress(vtable1), 1},
        {getCalleeAddress(vtable2), 2},
    };
    Record1.addValueData(IPVK_VTableTarget, 0, VD0, nullptr);
    Record1.addValueData(IPVK_VTableTarget, 1, VD2, nullptr);
  }

  Writer.addRecord(std::move(Record1), getProfWeight(), Err);
  Writer.addRecord({"callee1", 0x1235, {3, 4}}, Err);
  Writer.addRecord({"callee2", 0x1235, {3, 4}}, Err);
  Writer.addRecord({"callee3", 0x1235, {3, 4}}, Err);
  Writer.addRecord({"callee7", 0x1235, {3, 4}}, Err);
  Writer.addRecord({"callee8", 0x1235, {3, 4}}, Err);

  // Set writer value prof data endianness.
  Writer.setValueProfDataEndianness(getEndianness());

  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  // Set reader value prof data endianness.
  Reader->setValueProfDataEndianness(getEndianness());

  auto R = Reader->getInstrProfRecord("caller", 0x1234);
  ASSERT_THAT_ERROR(R.takeError(), Succeeded());

  // Test the number of instrumented indirect call sites and the number of
  // profiled values at each site.
  ASSERT_EQ(4U, R->getNumValueSites(IPVK_IndirectCallTarget));

  // Test the number of instrumented vtable sites and the number of profiled
  // values at each site.
  ASSERT_EQ(R->getNumValueSites(IPVK_VTableTarget), 2U);

  // First indirect site.
  {
    auto VD = R->getValueArrayForSite(IPVK_IndirectCallTarget, 0);
    ASSERT_THAT(VD, SizeIs(3));

    EXPECT_EQ(VD[0].Count, 3U * getProfWeight());
    EXPECT_EQ(VD[1].Count, 2U * getProfWeight());
    EXPECT_EQ(VD[2].Count, 1U * getProfWeight());

    EXPECT_STREQ((const char *)VD[0].Value, "callee3");
    EXPECT_STREQ((const char *)VD[1].Value, "callee2");
    EXPECT_STREQ((const char *)VD[2].Value, "callee1");
  }

  EXPECT_THAT(R->getValueArrayForSite(IPVK_IndirectCallTarget, 1), SizeIs(0));
  EXPECT_THAT(R->getValueArrayForSite(IPVK_IndirectCallTarget, 2), SizeIs(2));
  EXPECT_THAT(R->getValueArrayForSite(IPVK_IndirectCallTarget, 3), SizeIs(2));

  // First vtable site.
  {
    auto VD = R->getValueArrayForSite(IPVK_VTableTarget, 0);
    ASSERT_THAT(VD, SizeIs(3));

    EXPECT_EQ(VD[0].Count, 3U * getProfWeight());
    EXPECT_EQ(VD[1].Count, 2U * getProfWeight());
    EXPECT_EQ(VD[2].Count, 1U * getProfWeight());

    EXPECT_EQ(VD[0].Value, getCalleeAddress(vtable3));
    EXPECT_EQ(VD[1].Value, getCalleeAddress(vtable2));
    EXPECT_EQ(VD[2].Value, getCalleeAddress(vtable1));
  }

  // Second vtable site.
  {
    auto VD = R->getValueArrayForSite(IPVK_VTableTarget, 1);
    ASSERT_THAT(VD, SizeIs(2));

    EXPECT_EQ(VD[0].Count, 2U * getProfWeight());
    EXPECT_EQ(VD[1].Count, 1U * getProfWeight());

    EXPECT_EQ(VD[0].Value, getCalleeAddress(vtable2));
    EXPECT_EQ(VD[1].Value, getCalleeAddress(vtable1));
  }
}

INSTANTIATE_TEST_SUITE_P(
    WeightAndEndiannessTest, InstrProfReaderWriterTest,
    ::testing::Combine(
        ::testing::Bool(),          /* Sparse */
        ::testing::Values(1U, 10U), /* ProfWeight */
        ::testing::Values(llvm::endianness::big,
                          llvm::endianness::little) /* Endianness */
        ));

TEST_P(MaybeSparseInstrProfTest, annotate_vp_data) {
  NamedInstrProfRecord Record("caller", 0x1234, {1, 2});
  Record.reserveSites(IPVK_IndirectCallTarget, 1);
  InstrProfValueData VD0[] = {{1000, 1}, {2000, 2}, {3000, 3}, {5000, 5},
                              {4000, 4}, {6000, 6}};
  Record.addValueData(IPVK_IndirectCallTarget, 0, VD0, nullptr);
  Writer.addRecord(std::move(Record), Err);
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));
  auto R = Reader->getInstrProfRecord("caller", 0x1234);
  EXPECT_THAT_ERROR(R.takeError(), Succeeded());

  LLVMContext Ctx;
  std::unique_ptr<Module> M(new Module("MyModule", Ctx));
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(Ctx),
                                        /*isVarArg=*/false);
  Function *F =
      Function::Create(FTy, Function::ExternalLinkage, "caller", M.get());
  BasicBlock *BB = BasicBlock::Create(Ctx, "", F);

  IRBuilder<> Builder(BB);
  BasicBlock *TBB = BasicBlock::Create(Ctx, "", F);
  BasicBlock *FBB = BasicBlock::Create(Ctx, "", F);

  // Use branch instruction to annotate with value profile data for simplicity
  Instruction *Inst = Builder.CreateCondBr(Builder.getTrue(), TBB, FBB);
  Instruction *Inst2 = Builder.CreateCondBr(Builder.getTrue(), TBB, FBB);
  annotateValueSite(*M, *Inst, R.get(), IPVK_IndirectCallTarget, 0);

  uint64_t T;
  auto ValueData =
      getValueProfDataFromInst(*Inst, IPVK_IndirectCallTarget, 5, T);
  ASSERT_THAT(ValueData, SizeIs(3));
  ASSERT_EQ(21U, T);
  // The result should be sorted already:
  ASSERT_EQ(6000U, ValueData[0].Value);
  ASSERT_EQ(6U, ValueData[0].Count);
  ASSERT_EQ(5000U, ValueData[1].Value);
  ASSERT_EQ(5U, ValueData[1].Count);
  ASSERT_EQ(4000U, ValueData[2].Value);
  ASSERT_EQ(4U, ValueData[2].Count);
  ValueData = getValueProfDataFromInst(*Inst, IPVK_IndirectCallTarget, 1, T);
  ASSERT_THAT(ValueData, SizeIs(1));
  ASSERT_EQ(21U, T);

  ValueData = getValueProfDataFromInst(*Inst2, IPVK_IndirectCallTarget, 5, T);
  ASSERT_THAT(ValueData, SizeIs(0));

  // Remove the MD_prof metadata
  Inst->setMetadata(LLVMContext::MD_prof, 0);
  // Annotate 5 records this time.
  annotateValueSite(*M, *Inst, R.get(), IPVK_IndirectCallTarget, 0, 5);
  ValueData = getValueProfDataFromInst(*Inst, IPVK_IndirectCallTarget, 5, T);
  ASSERT_THAT(ValueData, SizeIs(5));
  ASSERT_EQ(21U, T);
  ASSERT_EQ(6000U, ValueData[0].Value);
  ASSERT_EQ(6U, ValueData[0].Count);
  ASSERT_EQ(5000U, ValueData[1].Value);
  ASSERT_EQ(5U, ValueData[1].Count);
  ASSERT_EQ(4000U, ValueData[2].Value);
  ASSERT_EQ(4U, ValueData[2].Count);
  ASSERT_EQ(3000U, ValueData[3].Value);
  ASSERT_EQ(3U, ValueData[3].Count);
  ASSERT_EQ(2000U, ValueData[4].Value);
  ASSERT_EQ(2U, ValueData[4].Count);

  // Remove the MD_prof metadata
  Inst->setMetadata(LLVMContext::MD_prof, 0);
  // Annotate with 4 records.
  InstrProfValueData VD0Sorted[] = {{1000, 6}, {2000, 5}, {3000, 4}, {4000, 3},
                              {5000, 2}, {6000, 1}};
  annotateValueSite(*M, *Inst, ArrayRef(VD0Sorted).slice(2), 10,
                    IPVK_IndirectCallTarget, 5);
  ValueData = getValueProfDataFromInst(*Inst, IPVK_IndirectCallTarget, 5, T);
  ASSERT_THAT(ValueData, SizeIs(4));
  ASSERT_EQ(10U, T);
  ASSERT_EQ(3000U, ValueData[0].Value);
  ASSERT_EQ(4U, ValueData[0].Count);
  ASSERT_EQ(4000U, ValueData[1].Value);
  ASSERT_EQ(3U, ValueData[1].Count);
  ASSERT_EQ(5000U, ValueData[2].Value);
  ASSERT_EQ(2U, ValueData[2].Count);
  ASSERT_EQ(6000U, ValueData[3].Value);
  ASSERT_EQ(1U, ValueData[3].Count);
}

TEST_P(MaybeSparseInstrProfTest, icall_and_vtable_data_merge) {
  static const char caller[] = "caller";
  NamedInstrProfRecord Record11(caller, 0x1234, {1, 2});
  NamedInstrProfRecord Record12(caller, 0x1234, {1, 2});

  // 5 value sites for indirect calls.
  {
    Record11.reserveSites(IPVK_IndirectCallTarget, 5);
    InstrProfValueData VD0[] = {{uint64_t(callee1), 1},
                                {uint64_t(callee2), 2},
                                {uint64_t(callee3), 3},
                                {uint64_t(callee4), 4}};
    Record11.addValueData(IPVK_IndirectCallTarget, 0, VD0, nullptr);

    // No value profile data at the second site.
    Record11.addValueData(IPVK_IndirectCallTarget, 1, {}, nullptr);

    InstrProfValueData VD2[] = {
        {uint64_t(callee1), 1}, {uint64_t(callee2), 2}, {uint64_t(callee3), 3}};
    Record11.addValueData(IPVK_IndirectCallTarget, 2, VD2, nullptr);

    InstrProfValueData VD3[] = {{uint64_t(callee7), 1}, {uint64_t(callee8), 2}};
    Record11.addValueData(IPVK_IndirectCallTarget, 3, VD3, nullptr);

    InstrProfValueData VD4[] = {
        {uint64_t(callee1), 1}, {uint64_t(callee2), 2}, {uint64_t(callee3), 3}};
    Record11.addValueData(IPVK_IndirectCallTarget, 4, VD4, nullptr);
  }
  // 3 value sites for vtables.
  {
    Record11.reserveSites(IPVK_VTableTarget, 3);
    InstrProfValueData VD0[] = {{getCalleeAddress(vtable1), 1},
                                {getCalleeAddress(vtable2), 2},
                                {getCalleeAddress(vtable3), 3},
                                {getCalleeAddress(vtable4), 4}};
    Record11.addValueData(IPVK_VTableTarget, 0, VD0, nullptr);

    InstrProfValueData VD2[] = {{getCalleeAddress(vtable1), 1},
                                {getCalleeAddress(vtable2), 2},
                                {getCalleeAddress(vtable3), 3}};
    Record11.addValueData(IPVK_VTableTarget, 1, VD2, nullptr);

    InstrProfValueData VD4[] = {{getCalleeAddress(vtable1), 1},
                                {getCalleeAddress(vtable2), 2},
                                {getCalleeAddress(vtable3), 3}};
    Record11.addValueData(IPVK_VTableTarget, 2, VD4, nullptr);
  }

  // A different record for the same caller.
  Record12.reserveSites(IPVK_IndirectCallTarget, 5);
  InstrProfValueData VD02[] = {{uint64_t(callee2), 5}, {uint64_t(callee3), 3}};
  Record12.addValueData(IPVK_IndirectCallTarget, 0, VD02, nullptr);

  // No value profile data at the second site.
  Record12.addValueData(IPVK_IndirectCallTarget, 1, {}, nullptr);

  InstrProfValueData VD22[] = {
      {uint64_t(callee2), 1}, {uint64_t(callee3), 3}, {uint64_t(callee4), 4}};
  Record12.addValueData(IPVK_IndirectCallTarget, 2, VD22, nullptr);

  Record12.addValueData(IPVK_IndirectCallTarget, 3, {}, nullptr);

  InstrProfValueData VD42[] = {
      {uint64_t(callee1), 1}, {uint64_t(callee2), 2}, {uint64_t(callee3), 3}};
  Record12.addValueData(IPVK_IndirectCallTarget, 4, VD42, nullptr);

  // 3 value sites for vtables.
  {
    Record12.reserveSites(IPVK_VTableTarget, 3);
    InstrProfValueData VD0[] = {{getCalleeAddress(vtable2), 5},
                                {getCalleeAddress(vtable3), 3}};
    Record12.addValueData(IPVK_VTableTarget, 0, VD0, nullptr);

    InstrProfValueData VD2[] = {{getCalleeAddress(vtable2), 1},
                                {getCalleeAddress(vtable3), 3},
                                {getCalleeAddress(vtable4), 4}};
    Record12.addValueData(IPVK_VTableTarget, 1, VD2, nullptr);

    InstrProfValueData VD4[] = {{getCalleeAddress(vtable1), 1},
                                {getCalleeAddress(vtable2), 2},
                                {getCalleeAddress(vtable3), 3}};
    Record12.addValueData(IPVK_VTableTarget, 2, VD4, nullptr);
  }

  Writer.addRecord(std::move(Record11), Err);
  // Merge profile data.
  Writer.addRecord(std::move(Record12), Err);

  Writer.addRecord({callee1, 0x1235, {3, 4}}, Err);
  Writer.addRecord({callee2, 0x1235, {3, 4}}, Err);
  Writer.addRecord({callee3, 0x1235, {3, 4}}, Err);
  Writer.addRecord({callee3, 0x1235, {3, 4}}, Err);
  Writer.addRecord({callee4, 0x1235, {3, 5}}, Err);
  Writer.addRecord({callee7, 0x1235, {3, 5}}, Err);
  Writer.addRecord({callee8, 0x1235, {3, 5}}, Err);
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  // Test the number of instrumented value sites and the number of profiled
  // values for each site.
  auto R = Reader->getInstrProfRecord("caller", 0x1234);
  EXPECT_THAT_ERROR(R.takeError(), Succeeded());
  // For indirect calls.
  ASSERT_EQ(5U, R->getNumValueSites(IPVK_IndirectCallTarget));
  // For vtables.
  ASSERT_EQ(R->getNumValueSites(IPVK_VTableTarget), 3U);

  // Test the merged values for indirect calls.
  {
    auto VD = R->getValueArrayForSite(IPVK_IndirectCallTarget, 0);
    ASSERT_THAT(VD, SizeIs(4));
    EXPECT_STREQ((const char *)VD[0].Value, "callee2");
    EXPECT_EQ(VD[0].Count, 7U);
    EXPECT_STREQ((const char *)VD[1].Value, "callee3");
    EXPECT_EQ(VD[1].Count, 6U);
    EXPECT_STREQ((const char *)VD[2].Value, "callee4");
    EXPECT_EQ(VD[2].Count, 4U);
    EXPECT_STREQ((const char *)VD[3].Value, "callee1");
    EXPECT_EQ(VD[3].Count, 1U);

    ASSERT_THAT(R->getValueArrayForSite(IPVK_IndirectCallTarget, 1), SizeIs(0));

    auto VD_2 = R->getValueArrayForSite(IPVK_IndirectCallTarget, 2);
    ASSERT_THAT(VD_2, SizeIs(4));
    EXPECT_STREQ((const char *)VD_2[0].Value, "callee3");
    EXPECT_EQ(VD_2[0].Count, 6U);
    EXPECT_STREQ((const char *)VD_2[1].Value, "callee4");
    EXPECT_EQ(VD_2[1].Count, 4U);
    EXPECT_STREQ((const char *)VD_2[2].Value, "callee2");
    EXPECT_EQ(VD_2[2].Count, 3U);
    EXPECT_STREQ((const char *)VD_2[3].Value, "callee1");
    EXPECT_EQ(VD_2[3].Count, 1U);

    auto VD_3 = R->getValueArrayForSite(IPVK_IndirectCallTarget, 3);
    ASSERT_THAT(VD_3, SizeIs(2));
    EXPECT_STREQ((const char *)VD_3[0].Value, "callee8");
    EXPECT_EQ(VD_3[0].Count, 2U);
    EXPECT_STREQ((const char *)VD_3[1].Value, "callee7");
    EXPECT_EQ(VD_3[1].Count, 1U);

    auto VD_4 = R->getValueArrayForSite(IPVK_IndirectCallTarget, 4);
    ASSERT_THAT(VD_4, SizeIs(3));
    EXPECT_STREQ((const char *)VD_4[0].Value, "callee3");
    EXPECT_EQ(VD_4[0].Count, 6U);
    EXPECT_STREQ((const char *)VD_4[1].Value, "callee2");
    EXPECT_EQ(VD_4[1].Count, 4U);
    EXPECT_STREQ((const char *)VD_4[2].Value, "callee1");
    EXPECT_EQ(VD_4[2].Count, 2U);
  }

  // Test the merged values for vtables
  {
    auto VD0 = R->getValueArrayForSite(IPVK_VTableTarget, 0);
    ASSERT_THAT(VD0, SizeIs(4));
    EXPECT_EQ(VD0[0].Value, getCalleeAddress(vtable2));
    EXPECT_EQ(VD0[0].Count, 7U);
    EXPECT_EQ(VD0[1].Value, getCalleeAddress(vtable3));
    EXPECT_EQ(VD0[1].Count, 6U);
    EXPECT_EQ(VD0[2].Value, getCalleeAddress(vtable4));
    EXPECT_EQ(VD0[2].Count, 4U);
    EXPECT_EQ(VD0[3].Value, getCalleeAddress(vtable1));
    EXPECT_EQ(VD0[3].Count, 1U);

    auto VD1 = R->getValueArrayForSite(IPVK_VTableTarget, 1);
    ASSERT_THAT(VD1, SizeIs(4));
    EXPECT_EQ(VD1[0].Value, getCalleeAddress(vtable3));
    EXPECT_EQ(VD1[0].Count, 6U);
    EXPECT_EQ(VD1[1].Value, getCalleeAddress(vtable4));
    EXPECT_EQ(VD1[1].Count, 4U);
    EXPECT_EQ(VD1[2].Value, getCalleeAddress(vtable2));
    EXPECT_EQ(VD1[2].Count, 3U);
    EXPECT_EQ(VD1[3].Value, getCalleeAddress(vtable1));
    EXPECT_EQ(VD1[3].Count, 1U);

    auto VD2 = R->getValueArrayForSite(IPVK_VTableTarget, 2);
    ASSERT_THAT(VD2, SizeIs(3));
    EXPECT_EQ(VD2[0].Value, getCalleeAddress(vtable3));
    EXPECT_EQ(VD2[0].Count, 6U);
    EXPECT_EQ(VD2[1].Value, getCalleeAddress(vtable2));
    EXPECT_EQ(VD2[1].Count, 4U);
    EXPECT_EQ(VD2[2].Value, getCalleeAddress(vtable1));
    EXPECT_EQ(VD2[2].Count, 2U);
  }
}

struct ValueProfileMergeEdgeCaseTest
    : public InstrProfTest,
      public ::testing::WithParamInterface<std::tuple<bool, uint32_t>> {
  void SetUp() override { Writer.setOutputSparse(std::get<0>(GetParam())); }

  uint32_t getValueProfileKind() const { return std::get<1>(GetParam()); }
};

TEST_P(ValueProfileMergeEdgeCaseTest, value_profile_data_merge_saturation) {
  const uint32_t ValueKind = getValueProfileKind();
  static const char bar[] = "bar";
  const uint64_t ProfiledValue = 0x5678;

  const uint64_t MaxValCount = std::numeric_limits<uint64_t>::max();
  const uint64_t MaxEdgeCount = getInstrMaxCountValue();

  instrprof_error Result;
  auto Err = [&](Error E) {
    Result = std::get<0>(InstrProfError::take(std::move(E)));
  };
  Result = instrprof_error::success;
  Writer.addRecord({"foo", 0x1234, {1}}, Err);
  ASSERT_EQ(Result, instrprof_error::success);

  // Verify counter overflow.
  Result = instrprof_error::success;
  Writer.addRecord({"foo", 0x1234, {MaxEdgeCount}}, Err);
  ASSERT_EQ(Result, instrprof_error::counter_overflow);

  Result = instrprof_error::success;
  Writer.addRecord({bar, 0x9012, {8}}, Err);
  ASSERT_EQ(Result, instrprof_error::success);

  NamedInstrProfRecord Record4("baz", 0x5678, {3, 4});
  Record4.reserveSites(ValueKind, 1);
  InstrProfValueData VD4[] = {{ProfiledValue, 1}};
  Record4.addValueData(ValueKind, 0, VD4, nullptr);
  Result = instrprof_error::success;
  Writer.addRecord(std::move(Record4), Err);
  ASSERT_EQ(Result, instrprof_error::success);

  // Verify value data counter overflow.
  NamedInstrProfRecord Record5("baz", 0x5678, {5, 6});
  Record5.reserveSites(ValueKind, 1);
  InstrProfValueData VD5[] = {{ProfiledValue, MaxValCount}};
  Record5.addValueData(ValueKind, 0, VD5, nullptr);
  Result = instrprof_error::success;
  Writer.addRecord(std::move(Record5), Err);
  ASSERT_EQ(Result, instrprof_error::counter_overflow);

  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  // Verify saturation of counts.
  auto ReadRecord1 = Reader->getInstrProfRecord("foo", 0x1234);
  ASSERT_THAT_ERROR(ReadRecord1.takeError(), Succeeded());
  EXPECT_EQ(MaxEdgeCount, ReadRecord1->Counts[0]);

  auto ReadRecord2 = Reader->getInstrProfRecord("baz", 0x5678);
  ASSERT_TRUE(bool(ReadRecord2));
  ASSERT_EQ(1U, ReadRecord2->getNumValueSites(ValueKind));
  auto VD = ReadRecord2->getValueArrayForSite(ValueKind, 0);
  EXPECT_EQ(ProfiledValue, VD[0].Value);
  EXPECT_EQ(MaxValCount, VD[0].Count);
}

// This test tests that when there are too many values for a given site, the
// merged results are properly truncated.
TEST_P(ValueProfileMergeEdgeCaseTest, value_profile_data_merge_site_trunc) {
  const uint32_t ValueKind = getValueProfileKind();
  static const char caller[] = "caller";

  NamedInstrProfRecord Record11(caller, 0x1234, {1, 2});
  NamedInstrProfRecord Record12(caller, 0x1234, {1, 2});

  // 2 value sites.
  Record11.reserveSites(ValueKind, 2);
  InstrProfValueData VD0[255];
  for (int I = 0; I < 255; I++) {
    VD0[I].Value = 2 * I;
    VD0[I].Count = 2 * I + 1000;
  }

  Record11.addValueData(ValueKind, 0, VD0, nullptr);
  Record11.addValueData(ValueKind, 1, {}, nullptr);

  Record12.reserveSites(ValueKind, 2);
  InstrProfValueData VD1[255];
  for (int I = 0; I < 255; I++) {
    VD1[I].Value = 2 * I + 1;
    VD1[I].Count = 2 * I + 1001;
  }

  Record12.addValueData(ValueKind, 0, VD1, nullptr);
  Record12.addValueData(ValueKind, 1, {}, nullptr);

  Writer.addRecord(std::move(Record11), Err);
  // Merge profile data.
  Writer.addRecord(std::move(Record12), Err);

  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  auto R = Reader->getInstrProfRecord("caller", 0x1234);
  ASSERT_THAT_ERROR(R.takeError(), Succeeded());
  ASSERT_EQ(2U, R->getNumValueSites(ValueKind));
  auto VD = R->getValueArrayForSite(ValueKind, 0);
  EXPECT_THAT(VD, SizeIs(255));
  for (unsigned I = 0; I < 255; I++) {
    EXPECT_EQ(VD[I].Value, 509 - I);
    EXPECT_EQ(VD[I].Count, 1509 - I);
  }
}

INSTANTIATE_TEST_SUITE_P(
    EdgeCaseTest, ValueProfileMergeEdgeCaseTest,
    ::testing::Combine(::testing::Bool(), /* Sparse */
                       ::testing::Values(IPVK_IndirectCallTarget,
                                         IPVK_MemOPSize,
                                         IPVK_VTableTarget) /* ValueKind */
                       ));

static void addValueProfData(InstrProfRecord &Record) {
  // Add test data for indirect calls.
  {
    Record.reserveSites(IPVK_IndirectCallTarget, 6);
    InstrProfValueData VD0[] = {{uint64_t(callee1), 400},
                                {uint64_t(callee2), 1000},
                                {uint64_t(callee3), 500},
                                {uint64_t(callee4), 300},
                                {uint64_t(callee5), 100}};
    Record.addValueData(IPVK_IndirectCallTarget, 0, VD0, nullptr);
    InstrProfValueData VD1[] = {{uint64_t(callee5), 800},
                                {uint64_t(callee3), 1000},
                                {uint64_t(callee2), 2500},
                                {uint64_t(callee1), 1300}};
    Record.addValueData(IPVK_IndirectCallTarget, 1, VD1, nullptr);
    InstrProfValueData VD2[] = {{uint64_t(callee6), 800},
                                {uint64_t(callee3), 1000},
                                {uint64_t(callee4), 5500}};
    Record.addValueData(IPVK_IndirectCallTarget, 2, VD2, nullptr);
    InstrProfValueData VD3[] = {{uint64_t(callee2), 1800},
                                {uint64_t(callee3), 2000}};
    Record.addValueData(IPVK_IndirectCallTarget, 3, VD3, nullptr);
    Record.addValueData(IPVK_IndirectCallTarget, 4, {}, nullptr);
    InstrProfValueData VD5[] = {{uint64_t(callee7), 1234},
                                {uint64_t(callee8), 5678}};
    Record.addValueData(IPVK_IndirectCallTarget, 5, VD5, nullptr);
  }

  // Add test data for vtables
  {
    Record.reserveSites(IPVK_VTableTarget, 4);
    InstrProfValueData VD0[] = {
        {getCalleeAddress(vtable1), 400}, {getCalleeAddress(vtable2), 1000},
        {getCalleeAddress(vtable3), 500}, {getCalleeAddress(vtable4), 300},
        {getCalleeAddress(vtable5), 100},
    };
    InstrProfValueData VD1[] = {{getCalleeAddress(vtable5), 800},
                                {getCalleeAddress(vtable3), 1000},
                                {getCalleeAddress(vtable2), 2500},
                                {getCalleeAddress(vtable1), 1300}};
    InstrProfValueData VD2[] = {
        {getCalleeAddress(vtable6), 800},
        {getCalleeAddress(vtable3), 1000},
        {getCalleeAddress(vtable4), 5500},
    };
    InstrProfValueData VD3[] = {{getCalleeAddress(vtable2), 1800},
                                {getCalleeAddress(vtable3), 2000}};
    Record.addValueData(IPVK_VTableTarget, 0, VD0, nullptr);
    Record.addValueData(IPVK_VTableTarget, 1, VD1, nullptr);
    Record.addValueData(IPVK_VTableTarget, 2, VD2, nullptr);
    Record.addValueData(IPVK_VTableTarget, 3, VD3, nullptr);
  }
}

TEST(ValueProfileReadWriteTest, value_prof_data_read_write) {
  InstrProfRecord SrcRecord({1ULL << 31, 2});
  addValueProfData(SrcRecord);
  std::unique_ptr<ValueProfData> VPData =
      ValueProfData::serializeFrom(SrcRecord);

  InstrProfRecord Record({1ULL << 31, 2});
  VPData->deserializeTo(Record, nullptr);

  // Now read data from Record and sanity check the data
  ASSERT_EQ(6U, Record.getNumValueSites(IPVK_IndirectCallTarget));

  auto Cmp = [](const InstrProfValueData &VD1, const InstrProfValueData &VD2) {
    return VD1.Count > VD2.Count;
  };

  SmallVector<InstrProfValueData> VD_0(
      Record.getValueArrayForSite(IPVK_IndirectCallTarget, 0));
  ASSERT_THAT(VD_0, SizeIs(5));
  llvm::sort(VD_0, Cmp);
  EXPECT_STREQ((const char *)VD_0[0].Value, "callee2");
  EXPECT_EQ(1000U, VD_0[0].Count);
  EXPECT_STREQ((const char *)VD_0[1].Value, "callee3");
  EXPECT_EQ(500U, VD_0[1].Count);
  EXPECT_STREQ((const char *)VD_0[2].Value, "callee1");
  EXPECT_EQ(400U, VD_0[2].Count);
  EXPECT_STREQ((const char *)VD_0[3].Value, "callee4");
  EXPECT_EQ(300U, VD_0[3].Count);
  EXPECT_STREQ((const char *)VD_0[4].Value, "callee5");
  EXPECT_EQ(100U, VD_0[4].Count);

  SmallVector<InstrProfValueData> VD_1(
      Record.getValueArrayForSite(IPVK_IndirectCallTarget, 1));
  ASSERT_THAT(VD_1, SizeIs(4));
  llvm::sort(VD_1, Cmp);
  EXPECT_STREQ((const char *)VD_1[0].Value, "callee2");
  EXPECT_EQ(VD_1[0].Count, 2500U);
  EXPECT_STREQ((const char *)VD_1[1].Value, "callee1");
  EXPECT_EQ(VD_1[1].Count, 1300U);
  EXPECT_STREQ((const char *)VD_1[2].Value, "callee3");
  EXPECT_EQ(VD_1[2].Count, 1000U);
  EXPECT_STREQ((const char *)VD_1[3].Value, "callee5");
  EXPECT_EQ(VD_1[3].Count, 800U);

  SmallVector<InstrProfValueData> VD_2(
      Record.getValueArrayForSite(IPVK_IndirectCallTarget, 2));
  ASSERT_THAT(VD_2, SizeIs(3));
  llvm::sort(VD_2, Cmp);
  EXPECT_STREQ((const char *)VD_2[0].Value, "callee4");
  EXPECT_EQ(VD_2[0].Count, 5500U);
  EXPECT_STREQ((const char *)VD_2[1].Value, "callee3");
  EXPECT_EQ(VD_2[1].Count, 1000U);
  EXPECT_STREQ((const char *)VD_2[2].Value, "callee6");
  EXPECT_EQ(VD_2[2].Count, 800U);

  SmallVector<InstrProfValueData> VD_3(
      Record.getValueArrayForSite(IPVK_IndirectCallTarget, 3));
  ASSERT_THAT(VD_3, SizeIs(2));
  llvm::sort(VD_3, Cmp);
  EXPECT_STREQ((const char *)VD_3[0].Value, "callee3");
  EXPECT_EQ(VD_3[0].Count, 2000U);
  EXPECT_STREQ((const char *)VD_3[1].Value, "callee2");
  EXPECT_EQ(VD_3[1].Count, 1800U);

  ASSERT_THAT(Record.getValueArrayForSite(IPVK_IndirectCallTarget, 4),
              SizeIs(0));
  ASSERT_THAT(Record.getValueArrayForSite(IPVK_IndirectCallTarget, 5),
              SizeIs(2));

  ASSERT_EQ(Record.getNumValueSites(IPVK_VTableTarget), 4U);

  SmallVector<InstrProfValueData> VD0(
      Record.getValueArrayForSite(IPVK_VTableTarget, 0));
  ASSERT_THAT(VD0, SizeIs(5));
  llvm::sort(VD0, Cmp);
  EXPECT_EQ(VD0[0].Value, getCalleeAddress(vtable2));
  EXPECT_EQ(VD0[0].Count, 1000U);
  EXPECT_EQ(VD0[1].Value, getCalleeAddress(vtable3));
  EXPECT_EQ(VD0[1].Count, 500U);
  EXPECT_EQ(VD0[2].Value, getCalleeAddress(vtable1));
  EXPECT_EQ(VD0[2].Count, 400U);
  EXPECT_EQ(VD0[3].Value, getCalleeAddress(vtable4));
  EXPECT_EQ(VD0[3].Count, 300U);
  EXPECT_EQ(VD0[4].Value, getCalleeAddress(vtable5));
  EXPECT_EQ(VD0[4].Count, 100U);

  SmallVector<InstrProfValueData> VD1(
      Record.getValueArrayForSite(IPVK_VTableTarget, 1));
  ASSERT_THAT(VD1, SizeIs(4));
  llvm::sort(VD1, Cmp);
  EXPECT_EQ(VD1[0].Value, getCalleeAddress(vtable2));
  EXPECT_EQ(VD1[0].Count, 2500U);
  EXPECT_EQ(VD1[1].Value, getCalleeAddress(vtable1));
  EXPECT_EQ(VD1[1].Count, 1300U);
  EXPECT_EQ(VD1[2].Value, getCalleeAddress(vtable3));
  EXPECT_EQ(VD1[2].Count, 1000U);
  EXPECT_EQ(VD1[3].Value, getCalleeAddress(vtable5));
  EXPECT_EQ(VD1[3].Count, 800U);

  SmallVector<InstrProfValueData> VD2(
      Record.getValueArrayForSite(IPVK_VTableTarget, 2));
  ASSERT_THAT(VD2, SizeIs(3));
  llvm::sort(VD2, Cmp);
  EXPECT_EQ(VD2[0].Value, getCalleeAddress(vtable4));
  EXPECT_EQ(VD2[0].Count, 5500U);
  EXPECT_EQ(VD2[1].Value, getCalleeAddress(vtable3));
  EXPECT_EQ(VD2[1].Count, 1000U);
  EXPECT_EQ(VD2[2].Value, getCalleeAddress(vtable6));
  EXPECT_EQ(VD2[2].Count, 800U);

  SmallVector<InstrProfValueData> VD3(
      Record.getValueArrayForSite(IPVK_VTableTarget, 3));
  ASSERT_THAT(VD3, SizeIs(2));
  llvm::sort(VD3, Cmp);
  EXPECT_EQ(VD3[0].Value, getCalleeAddress(vtable3));
  EXPECT_EQ(VD3[0].Count, 2000U);
  EXPECT_EQ(VD3[1].Value, getCalleeAddress(vtable2));
  EXPECT_EQ(VD3[1].Count, 1800U);
}

TEST(ValueProfileReadWriteTest, symtab_mapping) {
  NamedInstrProfRecord SrcRecord("caller", 0x1234, {1ULL << 31, 2});
  addValueProfData(SrcRecord);
  std::unique_ptr<ValueProfData> VPData =
      ValueProfData::serializeFrom(SrcRecord);

  NamedInstrProfRecord Record("caller", 0x1234, {1ULL << 31, 2});
  InstrProfSymtab Symtab;
  Symtab.mapAddress(uint64_t(callee1), 0x1000ULL);
  Symtab.mapAddress(uint64_t(callee2), 0x2000ULL);
  Symtab.mapAddress(uint64_t(callee3), 0x3000ULL);
  Symtab.mapAddress(uint64_t(callee4), 0x4000ULL);
  // Missing mapping for callee5

  auto getVTableStartAddr = [](const uint64_t *vtable) -> uint64_t {
    return uint64_t(vtable);
  };
  auto getVTableEndAddr = [](const uint64_t *vtable) -> uint64_t {
    return uint64_t(vtable) + 16;
  };
  auto getVTableMidAddr = [](const uint64_t *vtable) -> uint64_t {
    return uint64_t(vtable) + 8;
  };
  // vtable1, vtable2, vtable3, vtable4 get mapped; vtable5, vtable6 are not
  // mapped.
  Symtab.mapVTableAddress(getVTableStartAddr(vtable1),
                          getVTableEndAddr(vtable1), MD5Hash("vtable1"));
  Symtab.mapVTableAddress(getVTableStartAddr(vtable2),
                          getVTableEndAddr(vtable2), MD5Hash("vtable2"));
  Symtab.mapVTableAddress(getVTableStartAddr(vtable3),
                          getVTableEndAddr(vtable3), MD5Hash("vtable3"));
  Symtab.mapVTableAddress(getVTableStartAddr(vtable4),
                          getVTableEndAddr(vtable4), MD5Hash("vtable4"));

  VPData->deserializeTo(Record, &Symtab);

  // Now read data from Record and sanity check the data
  ASSERT_EQ(Record.getNumValueSites(IPVK_IndirectCallTarget), 6U);

  // Look up the value correpsonding to the middle of a vtable in symtab and
  // test that it's the hash of the name.
  EXPECT_EQ(Symtab.getVTableHashFromAddress(getVTableMidAddr(vtable1)),
            MD5Hash("vtable1"));
  EXPECT_EQ(Symtab.getVTableHashFromAddress(getVTableMidAddr(vtable2)),
            MD5Hash("vtable2"));
  EXPECT_EQ(Symtab.getVTableHashFromAddress(getVTableMidAddr(vtable3)),
            MD5Hash("vtable3"));
  EXPECT_EQ(Symtab.getVTableHashFromAddress(getVTableMidAddr(vtable4)),
            MD5Hash("vtable4"));

  auto Cmp = [](const InstrProfValueData &VD1, const InstrProfValueData &VD2) {
    return VD1.Count > VD2.Count;
  };
  SmallVector<InstrProfValueData> VD_0(
      Record.getValueArrayForSite(IPVK_IndirectCallTarget, 0));
  ASSERT_THAT(VD_0, SizeIs(5));
  llvm::sort(VD_0, Cmp);
  ASSERT_EQ(VD_0[0].Value, 0x2000ULL);
  ASSERT_EQ(VD_0[0].Count, 1000U);
  ASSERT_EQ(VD_0[1].Value, 0x3000ULL);
  ASSERT_EQ(VD_0[1].Count, 500U);
  ASSERT_EQ(VD_0[2].Value, 0x1000ULL);
  ASSERT_EQ(VD_0[2].Count, 400U);

  // callee5 does not have a mapped value -- default to 0.
  ASSERT_EQ(VD_0[4].Value, 0ULL);

  // Sanity check the vtable value data
  ASSERT_EQ(Record.getNumValueSites(IPVK_VTableTarget), 4U);

  {
    // The first vtable site.
    SmallVector<InstrProfValueData> VD(
        Record.getValueArrayForSite(IPVK_VTableTarget, 0));
    ASSERT_THAT(VD, SizeIs(5));
    llvm::sort(VD, Cmp);
    EXPECT_EQ(VD[0].Count, 1000U);
    EXPECT_EQ(VD[0].Value, MD5Hash("vtable2"));
    EXPECT_EQ(VD[1].Count, 500U);
    EXPECT_EQ(VD[1].Value, MD5Hash("vtable3"));
    EXPECT_EQ(VD[2].Value, MD5Hash("vtable1"));
    EXPECT_EQ(VD[2].Count, 400U);
    EXPECT_EQ(VD[3].Value, MD5Hash("vtable4"));
    EXPECT_EQ(VD[3].Count, 300U);

    // vtable5 isn't mapped -- default to 0.
    EXPECT_EQ(VD[4].Value, 0U);
    EXPECT_EQ(VD[4].Count, 100U);
  }

  {
    // The second vtable site.
    SmallVector<InstrProfValueData> VD(
        Record.getValueArrayForSite(IPVK_VTableTarget, 1));
    ASSERT_THAT(VD, SizeIs(4));
    llvm::sort(VD, Cmp);
    EXPECT_EQ(VD[0].Value, MD5Hash("vtable2"));
    EXPECT_EQ(VD[0].Count, 2500U);
    EXPECT_EQ(VD[1].Value, MD5Hash("vtable1"));
    EXPECT_EQ(VD[1].Count, 1300U);

    EXPECT_EQ(VD[2].Value, MD5Hash("vtable3"));
    EXPECT_EQ(VD[2].Count, 1000U);
    // vtable5 isn't mapped -- default to 0.
    EXPECT_EQ(VD[3].Value, 0U);
    EXPECT_EQ(VD[3].Count, 800U);
  }

  {
    // The third vtable site.
    SmallVector<InstrProfValueData> VD(
        Record.getValueArrayForSite(IPVK_VTableTarget, 2));
    ASSERT_THAT(VD, SizeIs(3));
    llvm::sort(VD, Cmp);
    EXPECT_EQ(VD[0].Count, 5500U);
    EXPECT_EQ(VD[0].Value, MD5Hash("vtable4"));
    EXPECT_EQ(VD[1].Count, 1000U);
    EXPECT_EQ(VD[1].Value, MD5Hash("vtable3"));
    // vtable6 isn't mapped -- default to 0.
    EXPECT_EQ(VD[2].Value, 0U);
    EXPECT_EQ(VD[2].Count, 800U);
  }

  {
    // The fourth vtable site.
    SmallVector<InstrProfValueData> VD(
        Record.getValueArrayForSite(IPVK_VTableTarget, 3));
    ASSERT_THAT(VD, SizeIs(2));
    llvm::sort(VD, Cmp);
    EXPECT_EQ(VD[0].Count, 2000U);
    EXPECT_EQ(VD[0].Value, MD5Hash("vtable3"));
    EXPECT_EQ(VD[1].Count, 1800U);
    EXPECT_EQ(VD[1].Value, MD5Hash("vtable2"));
  }
}

TEST_P(MaybeSparseInstrProfTest, get_max_function_count) {
  Writer.addRecord({"foo", 0x1234, {1ULL << 31, 2}}, Err);
  Writer.addRecord({"bar", 0, {1ULL << 63}}, Err);
  Writer.addRecord({"baz", 0x5678, {0, 0, 0, 0}}, Err);
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  ASSERT_EQ(1ULL << 63, Reader->getMaximumFunctionCount(/* IsCS */ false));
}

TEST_P(MaybeSparseInstrProfTest, get_weighted_function_counts) {
  Writer.addRecord({"foo", 0x1234, {1, 2}}, 3, Err);
  Writer.addRecord({"foo", 0x1235, {3, 4}}, 5, Err);
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  std::vector<uint64_t> Counts;
  EXPECT_THAT_ERROR(Reader->getFunctionCounts("foo", 0x1234, Counts),
                    Succeeded());
  ASSERT_EQ(2U, Counts.size());
  ASSERT_EQ(3U, Counts[0]);
  ASSERT_EQ(6U, Counts[1]);

  EXPECT_THAT_ERROR(Reader->getFunctionCounts("foo", 0x1235, Counts),
                    Succeeded());
  ASSERT_EQ(2U, Counts.size());
  ASSERT_EQ(15U, Counts[0]);
  ASSERT_EQ(20U, Counts[1]);
}

// Testing symtab creator interface used by indexed profile reader.
TEST(SymtabTest, instr_prof_symtab_test) {
  std::vector<StringRef> FuncNames;
  FuncNames.push_back("func1");
  FuncNames.push_back("func2");
  FuncNames.push_back("func3");
  FuncNames.push_back("bar1");
  FuncNames.push_back("bar2");
  FuncNames.push_back("bar3");
  InstrProfSymtab Symtab;
  EXPECT_THAT_ERROR(Symtab.create(FuncNames), Succeeded());
  StringRef R = Symtab.getFuncOrVarName(IndexedInstrProf::ComputeHash("func1"));
  ASSERT_EQ(StringRef("func1"), R);
  R = Symtab.getFuncOrVarName(IndexedInstrProf::ComputeHash("func2"));
  ASSERT_EQ(StringRef("func2"), R);
  R = Symtab.getFuncOrVarName(IndexedInstrProf::ComputeHash("func3"));
  ASSERT_EQ(StringRef("func3"), R);
  R = Symtab.getFuncOrVarName(IndexedInstrProf::ComputeHash("bar1"));
  ASSERT_EQ(StringRef("bar1"), R);
  R = Symtab.getFuncOrVarName(IndexedInstrProf::ComputeHash("bar2"));
  ASSERT_EQ(StringRef("bar2"), R);
  R = Symtab.getFuncOrVarName(IndexedInstrProf::ComputeHash("bar3"));
  ASSERT_EQ(StringRef("bar3"), R);

  // negative tests
  R = Symtab.getFuncOrVarName(IndexedInstrProf::ComputeHash("bar4"));
  ASSERT_EQ(StringRef(), R);
  R = Symtab.getFuncOrVarName(IndexedInstrProf::ComputeHash("foo4"));
  ASSERT_EQ(StringRef(), R);

  // Now incrementally update the symtab
  EXPECT_THAT_ERROR(Symtab.addFuncName("blah_1"), Succeeded());
  EXPECT_THAT_ERROR(Symtab.addFuncName("blah_2"), Succeeded());
  EXPECT_THAT_ERROR(Symtab.addFuncName("blah_3"), Succeeded());

  // Check again
  R = Symtab.getFuncOrVarName(IndexedInstrProf::ComputeHash("blah_1"));
  ASSERT_EQ(StringRef("blah_1"), R);
  R = Symtab.getFuncOrVarName(IndexedInstrProf::ComputeHash("blah_2"));
  ASSERT_EQ(StringRef("blah_2"), R);
  R = Symtab.getFuncOrVarName(IndexedInstrProf::ComputeHash("blah_3"));
  ASSERT_EQ(StringRef("blah_3"), R);
  R = Symtab.getFuncOrVarName(IndexedInstrProf::ComputeHash("func1"));
  ASSERT_EQ(StringRef("func1"), R);
  R = Symtab.getFuncOrVarName(IndexedInstrProf::ComputeHash("func2"));
  ASSERT_EQ(StringRef("func2"), R);
  R = Symtab.getFuncOrVarName(IndexedInstrProf::ComputeHash("func3"));
  ASSERT_EQ(StringRef("func3"), R);
  R = Symtab.getFuncOrVarName(IndexedInstrProf::ComputeHash("bar1"));
  ASSERT_EQ(StringRef("bar1"), R);
  R = Symtab.getFuncOrVarName(IndexedInstrProf::ComputeHash("bar2"));
  ASSERT_EQ(StringRef("bar2"), R);
  R = Symtab.getFuncOrVarName(IndexedInstrProf::ComputeHash("bar3"));
  ASSERT_EQ(StringRef("bar3"), R);
}

// Test that we get an error when creating a bogus symtab.
TEST(SymtabTest, instr_prof_bogus_symtab_empty_func_name) {
  InstrProfSymtab Symtab;
  EXPECT_TRUE(ErrorEquals(instrprof_error::malformed, Symtab.addFuncName("")));
}

// Testing symtab creator interface used by value profile transformer.
TEST(SymtabTest, instr_prof_symtab_module_test) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = std::make_unique<Module>("MyModule.cpp", Ctx);
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(Ctx),
                                        /*isVarArg=*/false);
  Function::Create(FTy, Function::ExternalLinkage, "Gfoo", M.get());
  Function::Create(FTy, Function::ExternalLinkage, "Gblah", M.get());
  Function::Create(FTy, Function::ExternalLinkage, "Gbar", M.get());
  Function::Create(FTy, Function::InternalLinkage, "Ifoo", M.get());
  Function::Create(FTy, Function::InternalLinkage, "Iblah", M.get());
  Function::Create(FTy, Function::InternalLinkage, "Ibar", M.get());
  Function::Create(FTy, Function::PrivateLinkage, "Pfoo", M.get());
  Function::Create(FTy, Function::PrivateLinkage, "Pblah", M.get());
  Function::Create(FTy, Function::PrivateLinkage, "Pbar", M.get());
  Function::Create(FTy, Function::WeakODRLinkage, "Wfoo", M.get());
  Function::Create(FTy, Function::WeakODRLinkage, "Wblah", M.get());
  Function::Create(FTy, Function::WeakODRLinkage, "Wbar", M.get());

  // [ptr, ptr, ptr]
  ArrayType *VTableArrayType = ArrayType::get(
      PointerType::get(Ctx, M->getDataLayout().getDefaultGlobalsAddressSpace()),
      3);
  Constant *Int32TyNull =
      llvm::ConstantExpr::getNullValue(PointerType::getUnqual(Ctx));
  SmallVector<llvm::Type *, 1> tys = {VTableArrayType};
  StructType *VTableType = llvm::StructType::get(Ctx, tys);

  // Create two vtables in the module, one with external linkage and the other
  // with local linkage.
  for (auto [Name, Linkage] :
       {std::pair{"ExternalGV", GlobalValue::ExternalLinkage},
        {"LocalGV", GlobalValue::InternalLinkage}}) {
    llvm::Twine FuncName(Name, StringRef("VFunc"));
    Function *VFunc = Function::Create(FTy, Linkage, FuncName, M.get());
    GlobalVariable *GV = new llvm::GlobalVariable(
        *M, VTableType, /* isConstant= */ true, Linkage,
        llvm::ConstantStruct::get(
            VTableType,
            {llvm::ConstantArray::get(VTableArrayType,
                                      {Int32TyNull, Int32TyNull, VFunc})}),
        Name);
    // Add type metadata for the test data, since vtables with type metadata
    // are added to symtab.
    GV->addTypeMetadata(16, MDString::get(Ctx, Name));
  }

  InstrProfSymtab ProfSymtab;
  EXPECT_THAT_ERROR(ProfSymtab.create(*M), Succeeded());

  StringRef Funcs[] = {"Gfoo", "Gblah", "Gbar", "Ifoo", "Iblah", "Ibar",
                       "Pfoo", "Pblah", "Pbar", "Wfoo", "Wblah", "Wbar"};

  for (unsigned I = 0; I < std::size(Funcs); I++) {
    Function *F = M->getFunction(Funcs[I]);

    std::string IRPGOName = getIRPGOFuncName(*F);
    auto IRPGOFuncName =
        ProfSymtab.getFuncOrVarName(IndexedInstrProf::ComputeHash(IRPGOName));
    EXPECT_EQ(IRPGOName, IRPGOFuncName);
    EXPECT_EQ(Funcs[I], getParsedIRPGOName(IRPGOFuncName).second);
    // Ensure we can still read this old record name.
    std::string PGOName = getPGOFuncName(*F);
    auto PGOFuncName =
        ProfSymtab.getFuncOrVarName(IndexedInstrProf::ComputeHash(PGOName));
    EXPECT_EQ(PGOName, PGOFuncName);
    EXPECT_THAT(PGOFuncName.str(), EndsWith(Funcs[I].str()));
  }

  for (auto [VTableName, PGOName] : {std::pair{"ExternalGV", "ExternalGV"},
                                     {"LocalGV", "MyModule.cpp;LocalGV"}}) {
    GlobalVariable *GV =
        M->getGlobalVariable(VTableName, /* AllowInternal=*/true);

    // Test that ProfSymtab returns the expected name given a hash.
    std::string IRPGOName = getPGOName(*GV);
    EXPECT_STREQ(IRPGOName.c_str(), PGOName);
    uint64_t GUID = IndexedInstrProf::ComputeHash(IRPGOName);
    EXPECT_EQ(IRPGOName, ProfSymtab.getFuncOrVarName(GUID));
    EXPECT_EQ(VTableName, getParsedIRPGOName(IRPGOName).second);

    // Test that ProfSymtab returns the expected global variable
    EXPECT_EQ(GV, ProfSymtab.getGlobalVariable(GUID));
  }
}

// Testing symtab serialization and creator/deserialization interface
// used by coverage map reader, and raw profile reader.
TEST(SymtabTest, instr_prof_symtab_compression_test) {
  std::vector<std::string> FuncNames1;
  std::vector<std::string> FuncNames2;
  for (int I = 0; I < 3; I++) {
    std::string str;
    raw_string_ostream OS(str);
    OS << "func_" << I;
    FuncNames1.push_back(OS.str());
    str.clear();
    OS << "f oooooooooooooo_" << I;
    FuncNames1.push_back(OS.str());
    str.clear();
    OS << "BAR_" << I;
    FuncNames2.push_back(OS.str());
    str.clear();
    OS << "BlahblahBlahblahBar_" << I;
    FuncNames2.push_back(OS.str());
  }

  for (bool DoCompression : {false, true}) {
    // Compressing:
    std::string FuncNameStrings1;
    EXPECT_THAT_ERROR(collectGlobalObjectNameStrings(
                          FuncNames1,
                          (DoCompression && compression::zlib::isAvailable()),
                          FuncNameStrings1),
                      Succeeded());

    // Compressing:
    std::string FuncNameStrings2;
    EXPECT_THAT_ERROR(collectGlobalObjectNameStrings(
                          FuncNames2,
                          (DoCompression && compression::zlib::isAvailable()),
                          FuncNameStrings2),
                      Succeeded());

    for (int Padding = 0; Padding < 2; Padding++) {
      // Join with paddings :
      std::string FuncNameStrings = FuncNameStrings1;
      for (int P = 0; P < Padding; P++) {
        FuncNameStrings.push_back('\0');
      }
      FuncNameStrings += FuncNameStrings2;

      // Now decompress:
      InstrProfSymtab Symtab;
      EXPECT_THAT_ERROR(Symtab.create(StringRef(FuncNameStrings)), Succeeded());

      // Now do the checks:
      // First sampling some data points:
      StringRef R =
          Symtab.getFuncOrVarName(IndexedInstrProf::ComputeHash(FuncNames1[0]));
      ASSERT_EQ(StringRef("func_0"), R);
      R = Symtab.getFuncOrVarName(IndexedInstrProf::ComputeHash(FuncNames1[1]));
      ASSERT_EQ(StringRef("f oooooooooooooo_0"), R);
      for (int I = 0; I < 3; I++) {
        std::string N[4];
        N[0] = FuncNames1[2 * I];
        N[1] = FuncNames1[2 * I + 1];
        N[2] = FuncNames2[2 * I];
        N[3] = FuncNames2[2 * I + 1];
        for (int J = 0; J < 4; J++) {
          StringRef R =
              Symtab.getFuncOrVarName(IndexedInstrProf::ComputeHash(N[J]));
          ASSERT_EQ(StringRef(N[J]), R);
        }
      }
    }
  }
}

TEST_P(MaybeSparseInstrProfTest, remapping_test) {
  Writer.addRecord({"_Z3fooi", 0x1234, {1, 2, 3, 4}}, Err);
  Writer.addRecord({"file;_Z3barf", 0x567, {5, 6, 7}}, Err);
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile), llvm::MemoryBuffer::getMemBuffer(R"(
    type i l
    name 3bar 4quux
  )"));

  std::vector<uint64_t> Counts;
  for (StringRef FooName : {"_Z3fooi", "_Z3fool"}) {
    EXPECT_THAT_ERROR(Reader->getFunctionCounts(FooName, 0x1234, Counts),
                      Succeeded());
    ASSERT_EQ(4u, Counts.size());
    EXPECT_EQ(1u, Counts[0]);
    EXPECT_EQ(2u, Counts[1]);
    EXPECT_EQ(3u, Counts[2]);
    EXPECT_EQ(4u, Counts[3]);
  }

  for (StringRef BarName : {"file;_Z3barf", "file;_Z4quuxf"}) {
    EXPECT_THAT_ERROR(Reader->getFunctionCounts(BarName, 0x567, Counts),
                      Succeeded());
    ASSERT_EQ(3u, Counts.size());
    EXPECT_EQ(5u, Counts[0]);
    EXPECT_EQ(6u, Counts[1]);
    EXPECT_EQ(7u, Counts[2]);
  }

  for (StringRef BadName : {"_Z3foof", "_Z4quuxi", "_Z3barl", "", "_ZZZ",
                            "_Z3barf", "otherfile:_Z4quuxf"}) {
    EXPECT_THAT_ERROR(Reader->getFunctionCounts(BadName, 0x1234, Counts),
                      Failed());
    EXPECT_THAT_ERROR(Reader->getFunctionCounts(BadName, 0x567, Counts),
                      Failed());
  }
}

TEST_F(SparseInstrProfTest, preserve_no_records) {
  Writer.addRecord({"foo", 0x1234, {0}}, Err);
  Writer.addRecord({"bar", 0x4321, {0, 0}}, Err);
  Writer.addRecord({"baz", 0x4321, {0, 0, 0}}, Err);

  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  auto I = Reader->begin(), E = Reader->end();
  ASSERT_TRUE(I == E);
}

INSTANTIATE_TEST_SUITE_P(MaybeSparse, MaybeSparseInstrProfTest,
                         ::testing::Bool());

#if defined(_LP64) && defined(EXPENSIVE_CHECKS)
TEST(ProfileReaderTest, ReadsLargeFiles) {
  const size_t LargeSize = 1ULL << 32; // 4GB

  auto RawProfile = WritableMemoryBuffer::getNewUninitMemBuffer(LargeSize);
  if (!RawProfile)
    GTEST_SKIP();
  auto RawProfileReaderOrErr = InstrProfReader::create(std::move(RawProfile));
  ASSERT_TRUE(
      std::get<0>(InstrProfError::take(RawProfileReaderOrErr.takeError())) ==
      instrprof_error::unrecognized_format);

  auto IndexedProfile = WritableMemoryBuffer::getNewUninitMemBuffer(LargeSize);
  if (!IndexedProfile)
    GTEST_SKIP();
  auto IndexedReaderOrErr =
      IndexedInstrProfReader::create(std::move(IndexedProfile), nullptr);
  ASSERT_TRUE(
      std::get<0>(InstrProfError::take(IndexedReaderOrErr.takeError())) ==
      instrprof_error::bad_magic);
}
#endif

} // end anonymous namespace
