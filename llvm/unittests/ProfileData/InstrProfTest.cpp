//===- unittest/ProfileData/InstrProfTest.cpp -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/ProfileData/InstrProfWriter.h"
#include "llvm/Support/Compression.h"
#include "gtest/gtest.h"
#include <cstdarg>

using namespace llvm;

static ::testing::AssertionResult NoError(Error E) {
  if (!E)
    return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << "error: " << toString(std::move(E))
                                       << "\n";
}

static ::testing::AssertionResult ErrorEquals(instrprof_error Expected,
                                              Error E) {
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

namespace {

struct InstrProfTest : ::testing::Test {
  InstrProfWriter Writer;
  std::unique_ptr<IndexedInstrProfReader> Reader;

  void SetUp() { Writer.setOutputSparse(false); }

  void readProfile(std::unique_ptr<MemoryBuffer> Profile) {
    auto ReaderOrErr = IndexedInstrProfReader::create(std::move(Profile));
    ASSERT_TRUE(NoError(ReaderOrErr.takeError()));
    Reader = std::move(ReaderOrErr.get());
  }
};

struct SparseInstrProfTest : public InstrProfTest {
  void SetUp() { Writer.setOutputSparse(true); }
};

struct MaybeSparseInstrProfTest : public InstrProfTest,
                                  public ::testing::WithParamInterface<bool> {
  void SetUp() { Writer.setOutputSparse(GetParam()); }
};

TEST_P(MaybeSparseInstrProfTest, write_and_read_empty_profile) {
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));
  ASSERT_TRUE(Reader->begin() == Reader->end());
}

TEST_P(MaybeSparseInstrProfTest, write_and_read_one_function) {
  InstrProfRecord Record("foo", 0x1234, {1, 2, 3, 4});
  NoError(Writer.addRecord(std::move(Record)));
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
  InstrProfRecord Record1("foo", 0x1234, {1, 2});
  InstrProfRecord Record2("foo", 0x1235, {3, 4});
  NoError(Writer.addRecord(std::move(Record1)));
  NoError(Writer.addRecord(std::move(Record2)));
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  Expected<InstrProfRecord> R = Reader->getInstrProfRecord("foo", 0x1234);
  ASSERT_TRUE(NoError(R.takeError()));
  ASSERT_EQ(2U, R->Counts.size());
  ASSERT_EQ(1U, R->Counts[0]);
  ASSERT_EQ(2U, R->Counts[1]);

  R = Reader->getInstrProfRecord("foo", 0x1235);
  ASSERT_TRUE(NoError(R.takeError()));
  ASSERT_EQ(2U, R->Counts.size());
  ASSERT_EQ(3U, R->Counts[0]);
  ASSERT_EQ(4U, R->Counts[1]);

  R = Reader->getInstrProfRecord("foo", 0x5678);
  ASSERT_TRUE(ErrorEquals(instrprof_error::hash_mismatch, R.takeError()));

  R = Reader->getInstrProfRecord("bar", 0x1234);
  ASSERT_TRUE(ErrorEquals(instrprof_error::unknown_function, R.takeError()));
}

TEST_P(MaybeSparseInstrProfTest, get_function_counts) {
  InstrProfRecord Record1("foo", 0x1234, {1, 2});
  InstrProfRecord Record2("foo", 0x1235, {3, 4});
  NoError(Writer.addRecord(std::move(Record1)));
  NoError(Writer.addRecord(std::move(Record2)));
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  std::vector<uint64_t> Counts;
  ASSERT_TRUE(NoError(Reader->getFunctionCounts("foo", 0x1234, Counts)));
  ASSERT_EQ(2U, Counts.size());
  ASSERT_EQ(1U, Counts[0]);
  ASSERT_EQ(2U, Counts[1]);

  ASSERT_TRUE(NoError(Reader->getFunctionCounts("foo", 0x1235, Counts)));
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
  InstrProfRecord Record1("func1", 0x1234, {97531});
  InstrProfRecord Record2("func2", 0x1234, {0, 0});
  InstrProfRecord Record3("func3", 0x1234,
                          {2305843009213693952, 1152921504606846976,
                           576460752303423488, 288230376151711744,
                           144115188075855872, 72057594037927936});
  InstrProfRecord Record4("func4", 0x1234, {0});
  NoError(Writer.addRecord(std::move(Record1)));
  NoError(Writer.addRecord(std::move(Record2)));
  NoError(Writer.addRecord(std::move(Record3)));
  NoError(Writer.addRecord(std::move(Record4)));
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  auto VerifySummary = [](ProfileSummary &IPS) mutable {
    ASSERT_EQ(ProfileSummary::PSK_Instr, IPS.getKind());
    ASSERT_EQ(2305843009213693952U, IPS.getMaxFunctionCount());
    ASSERT_EQ(2305843009213693952U, IPS.getMaxCount());
    ASSERT_EQ(10U, IPS.getNumCounts());
    ASSERT_EQ(4539628424389557499U, IPS.getTotalCount());
    std::vector<ProfileSummaryEntry> &Details = IPS.getDetailedSummary();
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
  ProfileSummary &PS = Reader->getSummary();
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
  M.setProfileSummary(MD);
  MD = M.getProfileSummary();
  ASSERT_TRUE(MD);
  PSFromMD = ProfileSummary::getFromMD(MD);
  ASSERT_TRUE(PSFromMD);
  VerifySummary(*PSFromMD);
  delete PSFromMD;
}

TEST_F(InstrProfTest, test_writer_merge) {
  InstrProfRecord Record1("func1", 0x1234, {42});
  NoError(Writer.addRecord(std::move(Record1)));

  InstrProfWriter Writer2;
  InstrProfRecord Record2("func2", 0x1234, {0, 0});
  NoError(Writer2.addRecord(std::move(Record2)));

  NoError(Writer.mergeRecordsFromWriter(std::move(Writer2)));

  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  Expected<InstrProfRecord> R = Reader->getInstrProfRecord("func1", 0x1234);
  ASSERT_TRUE(NoError(R.takeError()));
  ASSERT_EQ(1U, R->Counts.size());
  ASSERT_EQ(42U, R->Counts[0]);

  R = Reader->getInstrProfRecord("func2", 0x1234);
  ASSERT_TRUE(NoError(R.takeError()));
  ASSERT_EQ(2U, R->Counts.size());
  ASSERT_EQ(0U, R->Counts[0]);
  ASSERT_EQ(0U, R->Counts[1]);
}

static const char callee1[] = "callee1";
static const char callee2[] = "callee2";
static const char callee3[] = "callee3";
static const char callee4[] = "callee4";
static const char callee5[] = "callee5";
static const char callee6[] = "callee6";

TEST_P(MaybeSparseInstrProfTest, get_icall_data_read_write) {
  InstrProfRecord Record1("caller", 0x1234, {1, 2});
  InstrProfRecord Record2("callee1", 0x1235, {3, 4});
  InstrProfRecord Record3("callee2", 0x1235, {3, 4});
  InstrProfRecord Record4("callee3", 0x1235, {3, 4});

  // 4 value sites.
  Record1.reserveSites(IPVK_IndirectCallTarget, 4);
  InstrProfValueData VD0[] = {
      {(uint64_t)callee1, 1}, {(uint64_t)callee2, 2}, {(uint64_t)callee3, 3}};
  Record1.addValueData(IPVK_IndirectCallTarget, 0, VD0, 3, nullptr);
  // No value profile data at the second site.
  Record1.addValueData(IPVK_IndirectCallTarget, 1, nullptr, 0, nullptr);
  InstrProfValueData VD2[] = {{(uint64_t)callee1, 1}, {(uint64_t)callee2, 2}};
  Record1.addValueData(IPVK_IndirectCallTarget, 2, VD2, 2, nullptr);
  InstrProfValueData VD3[] = {{(uint64_t)callee1, 1}};
  Record1.addValueData(IPVK_IndirectCallTarget, 3, VD3, 1, nullptr);

  NoError(Writer.addRecord(std::move(Record1)));
  NoError(Writer.addRecord(std::move(Record2)));
  NoError(Writer.addRecord(std::move(Record3)));
  NoError(Writer.addRecord(std::move(Record4)));
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  Expected<InstrProfRecord> R = Reader->getInstrProfRecord("caller", 0x1234);
  ASSERT_TRUE(NoError(R.takeError()));
  ASSERT_EQ(4U, R->getNumValueSites(IPVK_IndirectCallTarget));
  ASSERT_EQ(3U, R->getNumValueDataForSite(IPVK_IndirectCallTarget, 0));
  ASSERT_EQ(0U, R->getNumValueDataForSite(IPVK_IndirectCallTarget, 1));
  ASSERT_EQ(2U, R->getNumValueDataForSite(IPVK_IndirectCallTarget, 2));
  ASSERT_EQ(1U, R->getNumValueDataForSite(IPVK_IndirectCallTarget, 3));

  uint64_t TotalC;
  std::unique_ptr<InstrProfValueData[]> VD =
      R->getValueForSite(IPVK_IndirectCallTarget, 0, &TotalC);

  ASSERT_EQ(3U, VD[0].Count);
  ASSERT_EQ(2U, VD[1].Count);
  ASSERT_EQ(1U, VD[2].Count);
  ASSERT_EQ(6U, TotalC);

  ASSERT_EQ(StringRef((const char *)VD[0].Value, 7), StringRef("callee3"));
  ASSERT_EQ(StringRef((const char *)VD[1].Value, 7), StringRef("callee2"));
  ASSERT_EQ(StringRef((const char *)VD[2].Value, 7), StringRef("callee1"));
}

TEST_P(MaybeSparseInstrProfTest, annotate_vp_data) {
  InstrProfRecord Record("caller", 0x1234, {1, 2});
  Record.reserveSites(IPVK_IndirectCallTarget, 1);
  InstrProfValueData VD0[] = {{1000, 1}, {2000, 2}, {3000, 3}, {5000, 5},
                              {4000, 4}, {6000, 6}};
  Record.addValueData(IPVK_IndirectCallTarget, 0, VD0, 6, nullptr);
  NoError(Writer.addRecord(std::move(Record)));
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));
  Expected<InstrProfRecord> R = Reader->getInstrProfRecord("caller", 0x1234);
  ASSERT_TRUE(NoError(R.takeError()));

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

  InstrProfValueData ValueData[5];
  uint32_t N;
  uint64_t T;
  bool Res = getValueProfDataFromInst(*Inst, IPVK_IndirectCallTarget, 5,
                                      ValueData, N, T);
  ASSERT_TRUE(Res);
  ASSERT_EQ(3U, N);
  ASSERT_EQ(21U, T);
  // The result should be sorted already:
  ASSERT_EQ(6000U, ValueData[0].Value);
  ASSERT_EQ(6U, ValueData[0].Count);
  ASSERT_EQ(5000U, ValueData[1].Value);
  ASSERT_EQ(5U, ValueData[1].Count);
  ASSERT_EQ(4000U, ValueData[2].Value);
  ASSERT_EQ(4U, ValueData[2].Count);
  Res = getValueProfDataFromInst(*Inst, IPVK_IndirectCallTarget, 1, ValueData,
                                 N, T);
  ASSERT_TRUE(Res);
  ASSERT_EQ(1U, N);
  ASSERT_EQ(21U, T);

  Res = getValueProfDataFromInst(*Inst2, IPVK_IndirectCallTarget, 5, ValueData,
                                 N, T);
  ASSERT_FALSE(Res);

  // Remove the MD_prof metadata 
  Inst->setMetadata(LLVMContext::MD_prof, 0);
  // Annotate 5 records this time.
  annotateValueSite(*M, *Inst, R.get(), IPVK_IndirectCallTarget, 0, 5);
  Res = getValueProfDataFromInst(*Inst, IPVK_IndirectCallTarget, 5,
                                      ValueData, N, T);
  ASSERT_TRUE(Res);
  ASSERT_EQ(5U, N);
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
  annotateValueSite(*M, *Inst, makeArrayRef(VD0Sorted).slice(2), 10,
                    IPVK_IndirectCallTarget, 5);
  Res = getValueProfDataFromInst(*Inst, IPVK_IndirectCallTarget, 5,
                                      ValueData, N, T);
  ASSERT_TRUE(Res);
  ASSERT_EQ(4U, N);
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

TEST_P(MaybeSparseInstrProfTest, get_icall_data_read_write_with_weight) {
  InstrProfRecord Record1("caller", 0x1234, {1, 2});
  InstrProfRecord Record2("callee1", 0x1235, {3, 4});
  InstrProfRecord Record3("callee2", 0x1235, {3, 4});
  InstrProfRecord Record4("callee3", 0x1235, {3, 4});

  // 4 value sites.
  Record1.reserveSites(IPVK_IndirectCallTarget, 4);
  InstrProfValueData VD0[] = {
      {(uint64_t)callee1, 1}, {(uint64_t)callee2, 2}, {(uint64_t)callee3, 3}};
  Record1.addValueData(IPVK_IndirectCallTarget, 0, VD0, 3, nullptr);
  // No value profile data at the second site.
  Record1.addValueData(IPVK_IndirectCallTarget, 1, nullptr, 0, nullptr);
  InstrProfValueData VD2[] = {{(uint64_t)callee1, 1}, {(uint64_t)callee2, 2}};
  Record1.addValueData(IPVK_IndirectCallTarget, 2, VD2, 2, nullptr);
  InstrProfValueData VD3[] = {{(uint64_t)callee1, 1}};
  Record1.addValueData(IPVK_IndirectCallTarget, 3, VD3, 1, nullptr);

  NoError(Writer.addRecord(std::move(Record1), 10));
  NoError(Writer.addRecord(std::move(Record2)));
  NoError(Writer.addRecord(std::move(Record3)));
  NoError(Writer.addRecord(std::move(Record4)));
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  Expected<InstrProfRecord> R = Reader->getInstrProfRecord("caller", 0x1234);
  ASSERT_TRUE(NoError(R.takeError()));
  ASSERT_EQ(4U, R->getNumValueSites(IPVK_IndirectCallTarget));
  ASSERT_EQ(3U, R->getNumValueDataForSite(IPVK_IndirectCallTarget, 0));
  ASSERT_EQ(0U, R->getNumValueDataForSite(IPVK_IndirectCallTarget, 1));
  ASSERT_EQ(2U, R->getNumValueDataForSite(IPVK_IndirectCallTarget, 2));
  ASSERT_EQ(1U, R->getNumValueDataForSite(IPVK_IndirectCallTarget, 3));

  uint64_t TotalC;
  std::unique_ptr<InstrProfValueData[]> VD =
      R->getValueForSite(IPVK_IndirectCallTarget, 0, &TotalC);
  ASSERT_EQ(30U, VD[0].Count);
  ASSERT_EQ(20U, VD[1].Count);
  ASSERT_EQ(10U, VD[2].Count);
  ASSERT_EQ(60U, TotalC);

  ASSERT_EQ(StringRef((const char *)VD[0].Value, 7), StringRef("callee3"));
  ASSERT_EQ(StringRef((const char *)VD[1].Value, 7), StringRef("callee2"));
  ASSERT_EQ(StringRef((const char *)VD[2].Value, 7), StringRef("callee1"));
}

TEST_P(MaybeSparseInstrProfTest, get_icall_data_read_write_big_endian) {
  InstrProfRecord Record1("caller", 0x1234, {1, 2});
  InstrProfRecord Record2("callee1", 0x1235, {3, 4});
  InstrProfRecord Record3("callee2", 0x1235, {3, 4});
  InstrProfRecord Record4("callee3", 0x1235, {3, 4});

  // 4 value sites.
  Record1.reserveSites(IPVK_IndirectCallTarget, 4);
  InstrProfValueData VD0[] = {
      {(uint64_t)callee1, 1}, {(uint64_t)callee2, 2}, {(uint64_t)callee3, 3}};
  Record1.addValueData(IPVK_IndirectCallTarget, 0, VD0, 3, nullptr);
  // No value profile data at the second site.
  Record1.addValueData(IPVK_IndirectCallTarget, 1, nullptr, 0, nullptr);
  InstrProfValueData VD2[] = {{(uint64_t)callee1, 1}, {(uint64_t)callee2, 2}};
  Record1.addValueData(IPVK_IndirectCallTarget, 2, VD2, 2, nullptr);
  InstrProfValueData VD3[] = {{(uint64_t)callee1, 1}};
  Record1.addValueData(IPVK_IndirectCallTarget, 3, VD3, 1, nullptr);

  NoError(Writer.addRecord(std::move(Record1)));
  NoError(Writer.addRecord(std::move(Record2)));
  NoError(Writer.addRecord(std::move(Record3)));
  NoError(Writer.addRecord(std::move(Record4)));

  // Set big endian output.
  Writer.setValueProfDataEndianness(support::big);

  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  // Set big endian input.
  Reader->setValueProfDataEndianness(support::big);

  Expected<InstrProfRecord> R = Reader->getInstrProfRecord("caller", 0x1234);
  ASSERT_TRUE(NoError(R.takeError()));
  ASSERT_EQ(4U, R->getNumValueSites(IPVK_IndirectCallTarget));
  ASSERT_EQ(3U, R->getNumValueDataForSite(IPVK_IndirectCallTarget, 0));
  ASSERT_EQ(0U, R->getNumValueDataForSite(IPVK_IndirectCallTarget, 1));
  ASSERT_EQ(2U, R->getNumValueDataForSite(IPVK_IndirectCallTarget, 2));
  ASSERT_EQ(1U, R->getNumValueDataForSite(IPVK_IndirectCallTarget, 3));

  std::unique_ptr<InstrProfValueData[]> VD =
      R->getValueForSite(IPVK_IndirectCallTarget, 0);
  ASSERT_EQ(StringRef((const char *)VD[0].Value, 7), StringRef("callee3"));
  ASSERT_EQ(StringRef((const char *)VD[1].Value, 7), StringRef("callee2"));
  ASSERT_EQ(StringRef((const char *)VD[2].Value, 7), StringRef("callee1"));

  // Restore little endian default:
  Writer.setValueProfDataEndianness(support::little);
}

TEST_P(MaybeSparseInstrProfTest, get_icall_data_merge1) {
  static const char caller[] = "caller";
  InstrProfRecord Record11(caller, 0x1234, {1, 2});
  InstrProfRecord Record12(caller, 0x1234, {1, 2});
  InstrProfRecord Record2(callee1, 0x1235, {3, 4});
  InstrProfRecord Record3(callee2, 0x1235, {3, 4});
  InstrProfRecord Record4(callee3, 0x1235, {3, 4});
  InstrProfRecord Record5(callee3, 0x1235, {3, 4});
  InstrProfRecord Record6(callee4, 0x1235, {3, 5});

  // 5 value sites.
  Record11.reserveSites(IPVK_IndirectCallTarget, 5);
  InstrProfValueData VD0[] = {{uint64_t(callee1), 1},
                              {uint64_t(callee2), 2},
                              {uint64_t(callee3), 3},
                              {uint64_t(callee4), 4}};
  Record11.addValueData(IPVK_IndirectCallTarget, 0, VD0, 4, nullptr);

  // No value profile data at the second site.
  Record11.addValueData(IPVK_IndirectCallTarget, 1, nullptr, 0, nullptr);

  InstrProfValueData VD2[] = {
      {uint64_t(callee1), 1}, {uint64_t(callee2), 2}, {uint64_t(callee3), 3}};
  Record11.addValueData(IPVK_IndirectCallTarget, 2, VD2, 3, nullptr);

  InstrProfValueData VD3[] = {{uint64_t(callee1), 1}};
  Record11.addValueData(IPVK_IndirectCallTarget, 3, VD3, 1, nullptr);

  InstrProfValueData VD4[] = {{uint64_t(callee1), 1},
                              {uint64_t(callee2), 2},
                              {uint64_t(callee3), 3}};
  Record11.addValueData(IPVK_IndirectCallTarget, 4, VD4, 3, nullptr);

  // A different record for the same caller.
  Record12.reserveSites(IPVK_IndirectCallTarget, 5);
  InstrProfValueData VD02[] = {{uint64_t(callee2), 5}, {uint64_t(callee3), 3}};
  Record12.addValueData(IPVK_IndirectCallTarget, 0, VD02, 2, nullptr);

  // No value profile data at the second site.
  Record12.addValueData(IPVK_IndirectCallTarget, 1, nullptr, 0, nullptr);

  InstrProfValueData VD22[] = {
      {uint64_t(callee2), 1}, {uint64_t(callee3), 3}, {uint64_t(callee4), 4}};
  Record12.addValueData(IPVK_IndirectCallTarget, 2, VD22, 3, nullptr);

  Record12.addValueData(IPVK_IndirectCallTarget, 3, nullptr, 0, nullptr);

  InstrProfValueData VD42[] = {{uint64_t(callee1), 1},
                               {uint64_t(callee2), 2},
                               {uint64_t(callee3), 3}};
  Record12.addValueData(IPVK_IndirectCallTarget, 4, VD42, 3, nullptr);

  NoError(Writer.addRecord(std::move(Record11)));
  // Merge profile data.
  NoError(Writer.addRecord(std::move(Record12)));

  NoError(Writer.addRecord(std::move(Record2)));
  NoError(Writer.addRecord(std::move(Record3)));
  NoError(Writer.addRecord(std::move(Record4)));
  NoError(Writer.addRecord(std::move(Record5)));
  NoError(Writer.addRecord(std::move(Record6)));
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  Expected<InstrProfRecord> R = Reader->getInstrProfRecord("caller", 0x1234);
  ASSERT_TRUE(NoError(R.takeError()));
  ASSERT_EQ(5U, R->getNumValueSites(IPVK_IndirectCallTarget));
  ASSERT_EQ(4U, R->getNumValueDataForSite(IPVK_IndirectCallTarget, 0));
  ASSERT_EQ(0U, R->getNumValueDataForSite(IPVK_IndirectCallTarget, 1));
  ASSERT_EQ(4U, R->getNumValueDataForSite(IPVK_IndirectCallTarget, 2));
  ASSERT_EQ(1U, R->getNumValueDataForSite(IPVK_IndirectCallTarget, 3));
  ASSERT_EQ(3U, R->getNumValueDataForSite(IPVK_IndirectCallTarget, 4));

  std::unique_ptr<InstrProfValueData[]> VD =
      R->getValueForSite(IPVK_IndirectCallTarget, 0);
  ASSERT_EQ(StringRef((const char *)VD[0].Value, 7), StringRef("callee2"));
  ASSERT_EQ(7U, VD[0].Count);
  ASSERT_EQ(StringRef((const char *)VD[1].Value, 7), StringRef("callee3"));
  ASSERT_EQ(6U, VD[1].Count);
  ASSERT_EQ(StringRef((const char *)VD[2].Value, 7), StringRef("callee4"));
  ASSERT_EQ(4U, VD[2].Count);
  ASSERT_EQ(StringRef((const char *)VD[3].Value, 7), StringRef("callee1"));
  ASSERT_EQ(1U, VD[3].Count);

  std::unique_ptr<InstrProfValueData[]> VD_2(
      R->getValueForSite(IPVK_IndirectCallTarget, 2));
  ASSERT_EQ(StringRef((const char *)VD_2[0].Value, 7), StringRef("callee3"));
  ASSERT_EQ(6U, VD_2[0].Count);
  ASSERT_EQ(StringRef((const char *)VD_2[1].Value, 7), StringRef("callee4"));
  ASSERT_EQ(4U, VD_2[1].Count);
  ASSERT_EQ(StringRef((const char *)VD_2[2].Value, 7), StringRef("callee2"));
  ASSERT_EQ(3U, VD_2[2].Count);
  ASSERT_EQ(StringRef((const char *)VD_2[3].Value, 7), StringRef("callee1"));
  ASSERT_EQ(1U, VD_2[3].Count);

  std::unique_ptr<InstrProfValueData[]> VD_3(
      R->getValueForSite(IPVK_IndirectCallTarget, 3));
  ASSERT_EQ(StringRef((const char *)VD_3[0].Value, 7), StringRef("callee1"));
  ASSERT_EQ(1U, VD_3[0].Count);

  std::unique_ptr<InstrProfValueData[]> VD_4(
      R->getValueForSite(IPVK_IndirectCallTarget, 4));
  ASSERT_EQ(StringRef((const char *)VD_4[0].Value, 7), StringRef("callee3"));
  ASSERT_EQ(6U, VD_4[0].Count);
  ASSERT_EQ(StringRef((const char *)VD_4[1].Value, 7), StringRef("callee2"));
  ASSERT_EQ(4U, VD_4[1].Count);
  ASSERT_EQ(StringRef((const char *)VD_4[2].Value, 7), StringRef("callee1"));
  ASSERT_EQ(2U, VD_4[2].Count);
}

TEST_P(MaybeSparseInstrProfTest, get_icall_data_merge1_saturation) {
  static const char bar[] = "bar";

  const uint64_t Max = std::numeric_limits<uint64_t>::max();

  InstrProfRecord Record1("foo", 0x1234, {1});
  auto Result1 = Writer.addRecord(std::move(Record1));
  ASSERT_EQ(InstrProfError::take(std::move(Result1)),
            instrprof_error::success);

  // Verify counter overflow.
  InstrProfRecord Record2("foo", 0x1234, {Max});
  auto Result2 = Writer.addRecord(std::move(Record2));
  ASSERT_EQ(InstrProfError::take(std::move(Result2)),
            instrprof_error::counter_overflow);

  InstrProfRecord Record3(bar, 0x9012, {8});
  auto Result3 = Writer.addRecord(std::move(Record3));
  ASSERT_EQ(InstrProfError::take(std::move(Result3)),
            instrprof_error::success);

  InstrProfRecord Record4("baz", 0x5678, {3, 4});
  Record4.reserveSites(IPVK_IndirectCallTarget, 1);
  InstrProfValueData VD4[] = {{uint64_t(bar), 1}};
  Record4.addValueData(IPVK_IndirectCallTarget, 0, VD4, 1, nullptr);
  auto Result4 = Writer.addRecord(std::move(Record4));
  ASSERT_EQ(InstrProfError::take(std::move(Result4)),
            instrprof_error::success);

  // Verify value data counter overflow.
  InstrProfRecord Record5("baz", 0x5678, {5, 6});
  Record5.reserveSites(IPVK_IndirectCallTarget, 1);
  InstrProfValueData VD5[] = {{uint64_t(bar), Max}};
  Record5.addValueData(IPVK_IndirectCallTarget, 0, VD5, 1, nullptr);
  auto Result5 = Writer.addRecord(std::move(Record5));
  ASSERT_EQ(InstrProfError::take(std::move(Result5)),
            instrprof_error::counter_overflow);

  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  // Verify saturation of counts.
  Expected<InstrProfRecord> ReadRecord1 =
      Reader->getInstrProfRecord("foo", 0x1234);
  ASSERT_TRUE(NoError(ReadRecord1.takeError()));
  ASSERT_EQ(Max, ReadRecord1->Counts[0]);

  Expected<InstrProfRecord> ReadRecord2 =
      Reader->getInstrProfRecord("baz", 0x5678);
  ASSERT_TRUE(bool(ReadRecord2));
  ASSERT_EQ(1U, ReadRecord2->getNumValueSites(IPVK_IndirectCallTarget));
  std::unique_ptr<InstrProfValueData[]> VD =
      ReadRecord2->getValueForSite(IPVK_IndirectCallTarget, 0);
  ASSERT_EQ(StringRef("bar"), StringRef((const char *)VD[0].Value, 3));
  ASSERT_EQ(Max, VD[0].Count);
}

// This test tests that when there are too many values
// for a given site, the merged results are properly
// truncated.
TEST_P(MaybeSparseInstrProfTest, get_icall_data_merge_site_trunc) {
  static const char caller[] = "caller";

  InstrProfRecord Record11(caller, 0x1234, {1, 2});
  InstrProfRecord Record12(caller, 0x1234, {1, 2});

  // 2 value sites.
  Record11.reserveSites(IPVK_IndirectCallTarget, 2);
  InstrProfValueData VD0[255];
  for (int I = 0; I < 255; I++) {
    VD0[I].Value = 2 * I;
    VD0[I].Count = 2 * I + 1000;
  }

  Record11.addValueData(IPVK_IndirectCallTarget, 0, VD0, 255, nullptr);
  Record11.addValueData(IPVK_IndirectCallTarget, 1, nullptr, 0, nullptr);

  Record12.reserveSites(IPVK_IndirectCallTarget, 2);
  InstrProfValueData VD1[255];
  for (int I = 0; I < 255; I++) {
    VD1[I].Value = 2 * I + 1;
    VD1[I].Count = 2 * I + 1001;
  }

  Record12.addValueData(IPVK_IndirectCallTarget, 0, VD1, 255, nullptr);
  Record12.addValueData(IPVK_IndirectCallTarget, 1, nullptr, 0, nullptr);

  NoError(Writer.addRecord(std::move(Record11)));
  // Merge profile data.
  NoError(Writer.addRecord(std::move(Record12)));

  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  Expected<InstrProfRecord> R = Reader->getInstrProfRecord("caller", 0x1234);
  ASSERT_TRUE(NoError(R.takeError()));
  std::unique_ptr<InstrProfValueData[]> VD(
      R->getValueForSite(IPVK_IndirectCallTarget, 0));
  ASSERT_EQ(2U, R->getNumValueSites(IPVK_IndirectCallTarget));
  ASSERT_EQ(255U, R->getNumValueDataForSite(IPVK_IndirectCallTarget, 0));
  for (unsigned I = 0; I < 255; I++) {
    ASSERT_EQ(VD[I].Value, 509 - I);
    ASSERT_EQ(VD[I].Count, 1509 - I);
  }
}

static void addValueProfData(InstrProfRecord &Record) {
  Record.reserveSites(IPVK_IndirectCallTarget, 5);
  InstrProfValueData VD0[] = {{uint64_t(callee1), 400},
                              {uint64_t(callee2), 1000},
                              {uint64_t(callee3), 500},
                              {uint64_t(callee4), 300},
                              {uint64_t(callee5), 100}};
  Record.addValueData(IPVK_IndirectCallTarget, 0, VD0, 5, nullptr);
  InstrProfValueData VD1[] = {{uint64_t(callee5), 800},
                              {uint64_t(callee3), 1000},
                              {uint64_t(callee2), 2500},
                              {uint64_t(callee1), 1300}};
  Record.addValueData(IPVK_IndirectCallTarget, 1, VD1, 4, nullptr);
  InstrProfValueData VD2[] = {{uint64_t(callee6), 800},
                              {uint64_t(callee3), 1000},
                              {uint64_t(callee4), 5500}};
  Record.addValueData(IPVK_IndirectCallTarget, 2, VD2, 3, nullptr);
  InstrProfValueData VD3[] = {{uint64_t(callee2), 1800},
                              {uint64_t(callee3), 2000}};
  Record.addValueData(IPVK_IndirectCallTarget, 3, VD3, 2, nullptr);
  Record.addValueData(IPVK_IndirectCallTarget, 4, nullptr, 0, nullptr);
}

TEST_P(MaybeSparseInstrProfTest, value_prof_data_read_write) {
  InstrProfRecord SrcRecord("caller", 0x1234, {1ULL << 31, 2});
  addValueProfData(SrcRecord);
  std::unique_ptr<ValueProfData> VPData =
      ValueProfData::serializeFrom(SrcRecord);

  InstrProfRecord Record("caller", 0x1234, {1ULL << 31, 2});
  VPData->deserializeTo(Record, nullptr);

  // Now read data from Record and sanity check the data
  ASSERT_EQ(5U, Record.getNumValueSites(IPVK_IndirectCallTarget));
  ASSERT_EQ(5U, Record.getNumValueDataForSite(IPVK_IndirectCallTarget, 0));
  ASSERT_EQ(4U, Record.getNumValueDataForSite(IPVK_IndirectCallTarget, 1));
  ASSERT_EQ(3U, Record.getNumValueDataForSite(IPVK_IndirectCallTarget, 2));
  ASSERT_EQ(2U, Record.getNumValueDataForSite(IPVK_IndirectCallTarget, 3));
  ASSERT_EQ(0U, Record.getNumValueDataForSite(IPVK_IndirectCallTarget, 4));

  auto Cmp = [](const InstrProfValueData &VD1, const InstrProfValueData &VD2) {
    return VD1.Count > VD2.Count;
  };
  std::unique_ptr<InstrProfValueData[]> VD_0(
      Record.getValueForSite(IPVK_IndirectCallTarget, 0));
  std::sort(&VD_0[0], &VD_0[5], Cmp);
  ASSERT_EQ(StringRef((const char *)VD_0[0].Value, 7), StringRef("callee2"));
  ASSERT_EQ(1000U, VD_0[0].Count);
  ASSERT_EQ(StringRef((const char *)VD_0[1].Value, 7), StringRef("callee3"));
  ASSERT_EQ(500U, VD_0[1].Count);
  ASSERT_EQ(StringRef((const char *)VD_0[2].Value, 7), StringRef("callee1"));
  ASSERT_EQ(400U, VD_0[2].Count);
  ASSERT_EQ(StringRef((const char *)VD_0[3].Value, 7), StringRef("callee4"));
  ASSERT_EQ(300U, VD_0[3].Count);
  ASSERT_EQ(StringRef((const char *)VD_0[4].Value, 7), StringRef("callee5"));
  ASSERT_EQ(100U, VD_0[4].Count);

  std::unique_ptr<InstrProfValueData[]> VD_1(
      Record.getValueForSite(IPVK_IndirectCallTarget, 1));
  std::sort(&VD_1[0], &VD_1[4], Cmp);
  ASSERT_EQ(StringRef((const char *)VD_1[0].Value, 7), StringRef("callee2"));
  ASSERT_EQ(2500U, VD_1[0].Count);
  ASSERT_EQ(StringRef((const char *)VD_1[1].Value, 7), StringRef("callee1"));
  ASSERT_EQ(1300U, VD_1[1].Count);
  ASSERT_EQ(StringRef((const char *)VD_1[2].Value, 7), StringRef("callee3"));
  ASSERT_EQ(1000U, VD_1[2].Count);
  ASSERT_EQ(StringRef((const char *)VD_1[3].Value, 7), StringRef("callee5"));
  ASSERT_EQ(800U, VD_1[3].Count);

  std::unique_ptr<InstrProfValueData[]> VD_2(
      Record.getValueForSite(IPVK_IndirectCallTarget, 2));
  std::sort(&VD_2[0], &VD_2[3], Cmp);
  ASSERT_EQ(StringRef((const char *)VD_2[0].Value, 7), StringRef("callee4"));
  ASSERT_EQ(5500U, VD_2[0].Count);
  ASSERT_EQ(StringRef((const char *)VD_2[1].Value, 7), StringRef("callee3"));
  ASSERT_EQ(1000U, VD_2[1].Count);
  ASSERT_EQ(StringRef((const char *)VD_2[2].Value, 7), StringRef("callee6"));
  ASSERT_EQ(800U, VD_2[2].Count);

  std::unique_ptr<InstrProfValueData[]> VD_3(
      Record.getValueForSite(IPVK_IndirectCallTarget, 3));
  std::sort(&VD_3[0], &VD_3[2], Cmp);
  ASSERT_EQ(StringRef((const char *)VD_3[0].Value, 7), StringRef("callee3"));
  ASSERT_EQ(2000U, VD_3[0].Count);
  ASSERT_EQ(StringRef((const char *)VD_3[1].Value, 7), StringRef("callee2"));
  ASSERT_EQ(1800U, VD_3[1].Count);
}

TEST_P(MaybeSparseInstrProfTest, value_prof_data_read_write_mapping) {

  InstrProfRecord SrcRecord("caller", 0x1234, {1ULL << 31, 2});
  addValueProfData(SrcRecord);
  std::unique_ptr<ValueProfData> VPData =
      ValueProfData::serializeFrom(SrcRecord);

  InstrProfRecord Record("caller", 0x1234, {1ULL << 31, 2});
  InstrProfSymtab Symtab;
  Symtab.mapAddress(uint64_t(callee1), 0x1000ULL);
  Symtab.mapAddress(uint64_t(callee2), 0x2000ULL);
  Symtab.mapAddress(uint64_t(callee3), 0x3000ULL);
  Symtab.mapAddress(uint64_t(callee4), 0x4000ULL);
  // Missing mapping for callee5
  Symtab.finalizeSymtab();

  VPData->deserializeTo(Record, &Symtab.getAddrHashMap());

  // Now read data from Record and sanity check the data
  ASSERT_EQ(5U, Record.getNumValueSites(IPVK_IndirectCallTarget));
  ASSERT_EQ(5U, Record.getNumValueDataForSite(IPVK_IndirectCallTarget, 0));

  auto Cmp = [](const InstrProfValueData &VD1, const InstrProfValueData &VD2) {
    return VD1.Count > VD2.Count;
  };
  std::unique_ptr<InstrProfValueData[]> VD_0(
      Record.getValueForSite(IPVK_IndirectCallTarget, 0));
  std::sort(&VD_0[0], &VD_0[5], Cmp);
  ASSERT_EQ(VD_0[0].Value, 0x2000ULL);
  ASSERT_EQ(1000U, VD_0[0].Count);
  ASSERT_EQ(VD_0[1].Value, 0x3000ULL);
  ASSERT_EQ(500U, VD_0[1].Count);
  ASSERT_EQ(VD_0[2].Value, 0x1000ULL);
  ASSERT_EQ(400U, VD_0[2].Count);

  // callee5 does not have a mapped value -- default to 0.
  ASSERT_EQ(VD_0[4].Value, 0ULL);
}

TEST_P(MaybeSparseInstrProfTest, get_max_function_count) {
  InstrProfRecord Record1("foo", 0x1234, {1ULL << 31, 2});
  InstrProfRecord Record2("bar", 0, {1ULL << 63});
  InstrProfRecord Record3("baz", 0x5678, {0, 0, 0, 0});
  NoError(Writer.addRecord(std::move(Record1)));
  NoError(Writer.addRecord(std::move(Record2)));
  NoError(Writer.addRecord(std::move(Record3)));
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  ASSERT_EQ(1ULL << 63, Reader->getMaximumFunctionCount());
}

TEST_P(MaybeSparseInstrProfTest, get_weighted_function_counts) {
  InstrProfRecord Record1("foo", 0x1234, {1, 2});
  InstrProfRecord Record2("foo", 0x1235, {3, 4});
  NoError(Writer.addRecord(std::move(Record1), 3));
  NoError(Writer.addRecord(std::move(Record2), 5));
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  std::vector<uint64_t> Counts;
  ASSERT_TRUE(NoError(Reader->getFunctionCounts("foo", 0x1234, Counts)));
  ASSERT_EQ(2U, Counts.size());
  ASSERT_EQ(3U, Counts[0]);
  ASSERT_EQ(6U, Counts[1]);

  ASSERT_TRUE(NoError(Reader->getFunctionCounts("foo", 0x1235, Counts)));
  ASSERT_EQ(2U, Counts.size());
  ASSERT_EQ(15U, Counts[0]);
  ASSERT_EQ(20U, Counts[1]);
}

// Testing symtab creator interface used by indexed profile reader.
TEST_P(MaybeSparseInstrProfTest, instr_prof_symtab_test) {
  std::vector<StringRef> FuncNames;
  FuncNames.push_back("func1");
  FuncNames.push_back("func2");
  FuncNames.push_back("func3");
  FuncNames.push_back("bar1");
  FuncNames.push_back("bar2");
  FuncNames.push_back("bar3");
  InstrProfSymtab Symtab;
  Symtab.create(FuncNames);
  StringRef R = Symtab.getFuncName(IndexedInstrProf::ComputeHash("func1"));
  ASSERT_EQ(StringRef("func1"), R);
  R = Symtab.getFuncName(IndexedInstrProf::ComputeHash("func2"));
  ASSERT_EQ(StringRef("func2"), R);
  R = Symtab.getFuncName(IndexedInstrProf::ComputeHash("func3"));
  ASSERT_EQ(StringRef("func3"), R);
  R = Symtab.getFuncName(IndexedInstrProf::ComputeHash("bar1"));
  ASSERT_EQ(StringRef("bar1"), R);
  R = Symtab.getFuncName(IndexedInstrProf::ComputeHash("bar2"));
  ASSERT_EQ(StringRef("bar2"), R);
  R = Symtab.getFuncName(IndexedInstrProf::ComputeHash("bar3"));
  ASSERT_EQ(StringRef("bar3"), R);

  // negative tests
  R = Symtab.getFuncName(IndexedInstrProf::ComputeHash("bar4"));
  ASSERT_EQ(StringRef(), R);
  R = Symtab.getFuncName(IndexedInstrProf::ComputeHash("foo4"));
  ASSERT_EQ(StringRef(), R);

  // Now incrementally update the symtab
  Symtab.addFuncName("blah_1");
  Symtab.addFuncName("blah_2");
  Symtab.addFuncName("blah_3");
  // Finalize it
  Symtab.finalizeSymtab();

  // Check again
  R = Symtab.getFuncName(IndexedInstrProf::ComputeHash("blah_1"));
  ASSERT_EQ(StringRef("blah_1"), R);
  R = Symtab.getFuncName(IndexedInstrProf::ComputeHash("blah_2"));
  ASSERT_EQ(StringRef("blah_2"), R);
  R = Symtab.getFuncName(IndexedInstrProf::ComputeHash("blah_3"));
  ASSERT_EQ(StringRef("blah_3"), R);
  R = Symtab.getFuncName(IndexedInstrProf::ComputeHash("func1"));
  ASSERT_EQ(StringRef("func1"), R);
  R = Symtab.getFuncName(IndexedInstrProf::ComputeHash("func2"));
  ASSERT_EQ(StringRef("func2"), R);
  R = Symtab.getFuncName(IndexedInstrProf::ComputeHash("func3"));
  ASSERT_EQ(StringRef("func3"), R);
  R = Symtab.getFuncName(IndexedInstrProf::ComputeHash("bar1"));
  ASSERT_EQ(StringRef("bar1"), R);
  R = Symtab.getFuncName(IndexedInstrProf::ComputeHash("bar2"));
  ASSERT_EQ(StringRef("bar2"), R);
  R = Symtab.getFuncName(IndexedInstrProf::ComputeHash("bar3"));
  ASSERT_EQ(StringRef("bar3"), R);
}

// Testing symtab creator interface used by value profile transformer.
TEST_P(MaybeSparseInstrProfTest, instr_prof_symtab_module_test) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = llvm::make_unique<Module>("MyModule.cpp", Ctx);
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

  InstrProfSymtab ProfSymtab;
  ProfSymtab.create(*M);

  StringRef Funcs[] = {"Gfoo", "Gblah", "Gbar", "Ifoo", "Iblah", "Ibar",
                       "Pfoo", "Pblah", "Pbar", "Wfoo", "Wblah", "Wbar"};

  for (unsigned I = 0; I < sizeof(Funcs) / sizeof(*Funcs); I++) {
    Function *F = M->getFunction(Funcs[I]);
    ASSERT_TRUE(F != nullptr);
    std::string PGOName = getPGOFuncName(*F);
    uint64_t Key = IndexedInstrProf::ComputeHash(PGOName);
    ASSERT_EQ(StringRef(PGOName),
              ProfSymtab.getFuncName(Key));
    ASSERT_EQ(StringRef(Funcs[I]), ProfSymtab.getOrigFuncName(Key));
  }
}

// Testing symtab serialization and creator/deserialization interface
// used by coverage map reader, and raw profile reader.
TEST_P(MaybeSparseInstrProfTest, instr_prof_symtab_compression_test) {
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
    NoError(collectPGOFuncNameStrings(
        FuncNames1, (DoCompression && zlib::isAvailable()), FuncNameStrings1));

    // Compressing:
    std::string FuncNameStrings2;
    NoError(collectPGOFuncNameStrings(
        FuncNames2, (DoCompression && zlib::isAvailable()), FuncNameStrings2));

    for (int Padding = 0; Padding < 2; Padding++) {
      // Join with paddings :
      std::string FuncNameStrings = FuncNameStrings1;
      for (int P = 0; P < Padding; P++) {
        FuncNameStrings.push_back('\0');
      }
      FuncNameStrings += FuncNameStrings2;

      // Now decompress:
      InstrProfSymtab Symtab;
      NoError(Symtab.create(StringRef(FuncNameStrings)));

      // Now do the checks:
      // First sampling some data points:
      StringRef R = Symtab.getFuncName(IndexedInstrProf::ComputeHash(FuncNames1[0]));
      ASSERT_EQ(StringRef("func_0"), R);
      R = Symtab.getFuncName(IndexedInstrProf::ComputeHash(FuncNames1[1]));
      ASSERT_EQ(StringRef("f oooooooooooooo_0"), R);
      for (int I = 0; I < 3; I++) {
        std::string N[4];
        N[0] = FuncNames1[2 * I];
        N[1] = FuncNames1[2 * I + 1];
        N[2] = FuncNames2[2 * I];
        N[3] = FuncNames2[2 * I + 1];
        for (int J = 0; J < 4; J++) {
          StringRef R = Symtab.getFuncName(IndexedInstrProf::ComputeHash(N[J]));
          ASSERT_EQ(StringRef(N[J]), R);
        }
      }
    }
  }
}

TEST_F(SparseInstrProfTest, preserve_no_records) {
  InstrProfRecord Record1("foo", 0x1234, {0});
  InstrProfRecord Record2("bar", 0x4321, {0, 0});
  InstrProfRecord Record3("bar", 0x4321, {0, 0, 0});

  NoError(Writer.addRecord(std::move(Record1)));
  NoError(Writer.addRecord(std::move(Record2)));
  NoError(Writer.addRecord(std::move(Record3)));
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  auto I = Reader->begin(), E = Reader->end();
  ASSERT_TRUE(I == E);
}

INSTANTIATE_TEST_CASE_P(MaybeSparse, MaybeSparseInstrProfTest,
                        ::testing::Bool(),);

} // end anonymous namespace
