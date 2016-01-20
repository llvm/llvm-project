//===- unittest/ProfileData/InstrProfTest.cpp -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/ProfileData/InstrProfWriter.h"
#include "llvm/Support/Compression.h"
#include "gtest/gtest.h"

#include <cstdarg>

using namespace llvm;

static ::testing::AssertionResult NoError(std::error_code EC) {
  if (!EC)
    return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << "error " << EC.value()
                                       << ": " << EC.message();
}

static ::testing::AssertionResult ErrorEquals(std::error_code Expected,
                                              std::error_code Found) {
  if (Expected == Found)
    return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << "error " << Found.value()
                                       << ": " << Found.message();
}

namespace {

struct InstrProfTest : ::testing::Test {
  InstrProfWriter Writer;
  std::unique_ptr<IndexedInstrProfReader> Reader;

  void readProfile(std::unique_ptr<MemoryBuffer> Profile) {
    auto ReaderOrErr = IndexedInstrProfReader::create(std::move(Profile));
    ASSERT_TRUE(NoError(ReaderOrErr.getError()));
    Reader = std::move(ReaderOrErr.get());
  }
};

TEST_F(InstrProfTest, write_and_read_empty_profile) {
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));
  ASSERT_TRUE(Reader->begin() == Reader->end());
}

TEST_F(InstrProfTest, write_and_read_one_function) {
  InstrProfRecord Record("foo", 0x1234, {1, 2, 3, 4});
  Writer.addRecord(std::move(Record));
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

TEST_F(InstrProfTest, get_instr_prof_record) {
  InstrProfRecord Record1("foo", 0x1234, {1, 2});
  InstrProfRecord Record2("foo", 0x1235, {3, 4});
  Writer.addRecord(std::move(Record1));
  Writer.addRecord(std::move(Record2));
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  ErrorOr<InstrProfRecord> R = Reader->getInstrProfRecord("foo", 0x1234);
  ASSERT_TRUE(NoError(R.getError()));
  ASSERT_EQ(2U, R.get().Counts.size());
  ASSERT_EQ(1U, R.get().Counts[0]);
  ASSERT_EQ(2U, R.get().Counts[1]);

  R = Reader->getInstrProfRecord("foo", 0x1235);
  ASSERT_TRUE(NoError(R.getError()));
  ASSERT_EQ(2U, R.get().Counts.size());
  ASSERT_EQ(3U, R.get().Counts[0]);
  ASSERT_EQ(4U, R.get().Counts[1]);

  R = Reader->getInstrProfRecord("foo", 0x5678);
  ASSERT_TRUE(ErrorEquals(instrprof_error::hash_mismatch, R.getError()));

  R = Reader->getInstrProfRecord("bar", 0x1234);
  ASSERT_TRUE(ErrorEquals(instrprof_error::unknown_function, R.getError()));
}

TEST_F(InstrProfTest, get_function_counts) {
  InstrProfRecord Record1("foo", 0x1234, {1, 2});
  InstrProfRecord Record2("foo", 0x1235, {3, 4});
  Writer.addRecord(std::move(Record1));
  Writer.addRecord(std::move(Record2));
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

  std::error_code EC;
  EC = Reader->getFunctionCounts("foo", 0x5678, Counts);
  ASSERT_TRUE(ErrorEquals(instrprof_error::hash_mismatch, EC));

  EC = Reader->getFunctionCounts("bar", 0x1234, Counts);
  ASSERT_TRUE(ErrorEquals(instrprof_error::unknown_function, EC));
}

TEST_F(InstrProfTest, get_icall_data_read_write) {
  InstrProfRecord Record1("caller", 0x1234, {1, 2});
  InstrProfRecord Record2("callee1", 0x1235, {3, 4});
  InstrProfRecord Record3("callee2", 0x1235, {3, 4});
  InstrProfRecord Record4("callee3", 0x1235, {3, 4});

  // 4 value sites.
  Record1.reserveSites(IPVK_IndirectCallTarget, 4);
  InstrProfValueData VD0[] = {{(uint64_t) "callee1", 1},
                              {(uint64_t) "callee2", 2},
                              {(uint64_t) "callee3", 3}};
  Record1.addValueData(IPVK_IndirectCallTarget, 0, VD0, 3, nullptr);
  // No value profile data at the second site.
  Record1.addValueData(IPVK_IndirectCallTarget, 1, nullptr, 0, nullptr);
  InstrProfValueData VD2[] = {{(uint64_t) "callee1", 1},
                              {(uint64_t) "callee2", 2}};
  Record1.addValueData(IPVK_IndirectCallTarget, 2, VD2, 2, nullptr);
  InstrProfValueData VD3[] = {{(uint64_t) "callee1", 1}};
  Record1.addValueData(IPVK_IndirectCallTarget, 3, VD3, 1, nullptr);

  Writer.addRecord(std::move(Record1));
  Writer.addRecord(std::move(Record2));
  Writer.addRecord(std::move(Record3));
  Writer.addRecord(std::move(Record4));
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  ErrorOr<InstrProfRecord> R = Reader->getInstrProfRecord("caller", 0x1234);
  ASSERT_TRUE(NoError(R.getError()));
  ASSERT_EQ(4U, R.get().getNumValueSites(IPVK_IndirectCallTarget));
  ASSERT_EQ(3U, R.get().getNumValueDataForSite(IPVK_IndirectCallTarget, 0));
  ASSERT_EQ(0U, R.get().getNumValueDataForSite(IPVK_IndirectCallTarget, 1));
  ASSERT_EQ(2U, R.get().getNumValueDataForSite(IPVK_IndirectCallTarget, 2));
  ASSERT_EQ(1U, R.get().getNumValueDataForSite(IPVK_IndirectCallTarget, 3));

  std::unique_ptr<InstrProfValueData[]> VD =
      R.get().getValueForSite(IPVK_IndirectCallTarget, 0);

  ASSERT_EQ(3U, VD[0].Count);
  ASSERT_EQ(2U, VD[1].Count);
  ASSERT_EQ(1U, VD[2].Count);

  ASSERT_EQ(StringRef((const char *)VD[0].Value, 7), StringRef("callee3"));
  ASSERT_EQ(StringRef((const char *)VD[1].Value, 7), StringRef("callee2"));
  ASSERT_EQ(StringRef((const char *)VD[2].Value, 7), StringRef("callee1"));
}

TEST_F(InstrProfTest, get_icall_data_read_write_with_weight) {
  InstrProfRecord Record1("caller", 0x1234, {1, 2});
  InstrProfRecord Record2("callee1", 0x1235, {3, 4});
  InstrProfRecord Record3("callee2", 0x1235, {3, 4});
  InstrProfRecord Record4("callee3", 0x1235, {3, 4});

  // 4 value sites.
  Record1.reserveSites(IPVK_IndirectCallTarget, 4);
  InstrProfValueData VD0[] = {{(uint64_t) "callee1", 1},
                              {(uint64_t) "callee2", 2},
                              {(uint64_t) "callee3", 3}};
  Record1.addValueData(IPVK_IndirectCallTarget, 0, VD0, 3, nullptr);
  // No value profile data at the second site.
  Record1.addValueData(IPVK_IndirectCallTarget, 1, nullptr, 0, nullptr);
  InstrProfValueData VD2[] = {{(uint64_t) "callee1", 1},
                              {(uint64_t) "callee2", 2}};
  Record1.addValueData(IPVK_IndirectCallTarget, 2, VD2, 2, nullptr);
  InstrProfValueData VD3[] = {{(uint64_t) "callee1", 1}};
  Record1.addValueData(IPVK_IndirectCallTarget, 3, VD3, 1, nullptr);

  Writer.addRecord(std::move(Record1), 10);
  Writer.addRecord(std::move(Record2));
  Writer.addRecord(std::move(Record3));
  Writer.addRecord(std::move(Record4));
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  ErrorOr<InstrProfRecord> R = Reader->getInstrProfRecord("caller", 0x1234);
  ASSERT_TRUE(NoError(R.getError()));
  ASSERT_EQ(4U, R.get().getNumValueSites(IPVK_IndirectCallTarget));
  ASSERT_EQ(3U, R.get().getNumValueDataForSite(IPVK_IndirectCallTarget, 0));
  ASSERT_EQ(0U, R.get().getNumValueDataForSite(IPVK_IndirectCallTarget, 1));
  ASSERT_EQ(2U, R.get().getNumValueDataForSite(IPVK_IndirectCallTarget, 2));
  ASSERT_EQ(1U, R.get().getNumValueDataForSite(IPVK_IndirectCallTarget, 3));

  std::unique_ptr<InstrProfValueData[]> VD =
      R.get().getValueForSite(IPVK_IndirectCallTarget, 0);
  ASSERT_EQ(30U, VD[0].Count);
  ASSERT_EQ(20U, VD[1].Count);
  ASSERT_EQ(10U, VD[2].Count);

  ASSERT_EQ(StringRef((const char *)VD[0].Value, 7), StringRef("callee3"));
  ASSERT_EQ(StringRef((const char *)VD[1].Value, 7), StringRef("callee2"));
  ASSERT_EQ(StringRef((const char *)VD[2].Value, 7), StringRef("callee1"));
}

TEST_F(InstrProfTest, get_icall_data_read_write_big_endian) {
  InstrProfRecord Record1("caller", 0x1234, {1, 2});
  InstrProfRecord Record2("callee1", 0x1235, {3, 4});
  InstrProfRecord Record3("callee2", 0x1235, {3, 4});
  InstrProfRecord Record4("callee3", 0x1235, {3, 4});

  // 4 value sites.
  Record1.reserveSites(IPVK_IndirectCallTarget, 4);
  InstrProfValueData VD0[] = {{(uint64_t) "callee1", 1},
                              {(uint64_t) "callee2", 2},
                              {(uint64_t) "callee3", 3}};
  Record1.addValueData(IPVK_IndirectCallTarget, 0, VD0, 3, nullptr);
  // No value profile data at the second site.
  Record1.addValueData(IPVK_IndirectCallTarget, 1, nullptr, 0, nullptr);
  InstrProfValueData VD2[] = {{(uint64_t) "callee1", 1},
                              {(uint64_t) "callee2", 2}};
  Record1.addValueData(IPVK_IndirectCallTarget, 2, VD2, 2, nullptr);
  InstrProfValueData VD3[] = {{(uint64_t) "callee1", 1}};
  Record1.addValueData(IPVK_IndirectCallTarget, 3, VD3, 1, nullptr);

  Writer.addRecord(std::move(Record1));
  Writer.addRecord(std::move(Record2));
  Writer.addRecord(std::move(Record3));
  Writer.addRecord(std::move(Record4));

  // Set big endian output.
  Writer.setValueProfDataEndianness(support::big);

  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  // Set big endian input.
  Reader->setValueProfDataEndianness(support::big);

  ErrorOr<InstrProfRecord> R = Reader->getInstrProfRecord("caller", 0x1234);
  ASSERT_TRUE(NoError(R.getError()));
  ASSERT_EQ(4U, R.get().getNumValueSites(IPVK_IndirectCallTarget));
  ASSERT_EQ(3U, R.get().getNumValueDataForSite(IPVK_IndirectCallTarget, 0));
  ASSERT_EQ(0U, R.get().getNumValueDataForSite(IPVK_IndirectCallTarget, 1));
  ASSERT_EQ(2U, R.get().getNumValueDataForSite(IPVK_IndirectCallTarget, 2));
  ASSERT_EQ(1U, R.get().getNumValueDataForSite(IPVK_IndirectCallTarget, 3));

  std::unique_ptr<InstrProfValueData[]> VD =
      R.get().getValueForSite(IPVK_IndirectCallTarget, 0);
  ASSERT_EQ(StringRef((const char *)VD[0].Value, 7), StringRef("callee3"));
  ASSERT_EQ(StringRef((const char *)VD[1].Value, 7), StringRef("callee2"));
  ASSERT_EQ(StringRef((const char *)VD[2].Value, 7), StringRef("callee1"));

  // Restore little endian default:
  Writer.setValueProfDataEndianness(support::little);
}

TEST_F(InstrProfTest, get_icall_data_merge1) {
  static const char caller[] = "caller";
  static const char callee1[] = "callee1";
  static const char callee2[] = "callee2";
  static const char callee3[] = "callee3";
  static const char callee4[] = "callee4";

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

  // A differnt record for the same caller.
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

  Writer.addRecord(std::move(Record11));
  // Merge profile data.
  Writer.addRecord(std::move(Record12));

  Writer.addRecord(std::move(Record2));
  Writer.addRecord(std::move(Record3));
  Writer.addRecord(std::move(Record4));
  Writer.addRecord(std::move(Record5));
  Writer.addRecord(std::move(Record6));
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  ErrorOr<InstrProfRecord> R = Reader->getInstrProfRecord("caller", 0x1234);
  ASSERT_TRUE(NoError(R.getError()));
  ASSERT_EQ(5U, R.get().getNumValueSites(IPVK_IndirectCallTarget));
  ASSERT_EQ(4U, R.get().getNumValueDataForSite(IPVK_IndirectCallTarget, 0));
  ASSERT_EQ(0U, R.get().getNumValueDataForSite(IPVK_IndirectCallTarget, 1));
  ASSERT_EQ(4U, R.get().getNumValueDataForSite(IPVK_IndirectCallTarget, 2));
  ASSERT_EQ(1U, R.get().getNumValueDataForSite(IPVK_IndirectCallTarget, 3));
  ASSERT_EQ(3U, R.get().getNumValueDataForSite(IPVK_IndirectCallTarget, 4));

  std::unique_ptr<InstrProfValueData[]> VD =
      R.get().getValueForSite(IPVK_IndirectCallTarget, 0);
  ASSERT_EQ(StringRef((const char *)VD[0].Value, 7), StringRef("callee2"));
  ASSERT_EQ(7U, VD[0].Count);
  ASSERT_EQ(StringRef((const char *)VD[1].Value, 7), StringRef("callee3"));
  ASSERT_EQ(6U, VD[1].Count);
  ASSERT_EQ(StringRef((const char *)VD[2].Value, 7), StringRef("callee4"));
  ASSERT_EQ(4U, VD[2].Count);
  ASSERT_EQ(StringRef((const char *)VD[3].Value, 7), StringRef("callee1"));
  ASSERT_EQ(1U, VD[3].Count);

  std::unique_ptr<InstrProfValueData[]> VD_2(
      R.get().getValueForSite(IPVK_IndirectCallTarget, 2));
  ASSERT_EQ(StringRef((const char *)VD_2[0].Value, 7), StringRef("callee3"));
  ASSERT_EQ(6U, VD_2[0].Count);
  ASSERT_EQ(StringRef((const char *)VD_2[1].Value, 7), StringRef("callee4"));
  ASSERT_EQ(4U, VD_2[1].Count);
  ASSERT_EQ(StringRef((const char *)VD_2[2].Value, 7), StringRef("callee2"));
  ASSERT_EQ(3U, VD_2[2].Count);
  ASSERT_EQ(StringRef((const char *)VD_2[3].Value, 7), StringRef("callee1"));
  ASSERT_EQ(1U, VD_2[3].Count);

  std::unique_ptr<InstrProfValueData[]> VD_3(
      R.get().getValueForSite(IPVK_IndirectCallTarget, 3));
  ASSERT_EQ(StringRef((const char *)VD_3[0].Value, 7), StringRef("callee1"));
  ASSERT_EQ(1U, VD_3[0].Count);

  std::unique_ptr<InstrProfValueData[]> VD_4(
      R.get().getValueForSite(IPVK_IndirectCallTarget, 4));
  ASSERT_EQ(StringRef((const char *)VD_4[0].Value, 7), StringRef("callee3"));
  ASSERT_EQ(6U, VD_4[0].Count);
  ASSERT_EQ(StringRef((const char *)VD_4[1].Value, 7), StringRef("callee2"));
  ASSERT_EQ(4U, VD_4[1].Count);
  ASSERT_EQ(StringRef((const char *)VD_4[2].Value, 7), StringRef("callee1"));
  ASSERT_EQ(2U, VD_4[2].Count);
}

TEST_F(InstrProfTest, get_icall_data_merge1_saturation) {
  static const char bar[] = "bar";

  const uint64_t Max = std::numeric_limits<uint64_t>::max();

  InstrProfRecord Record1("foo", 0x1234, {1});
  auto Result1 = Writer.addRecord(std::move(Record1));
  ASSERT_EQ(Result1, instrprof_error::success);

  // Verify counter overflow.
  InstrProfRecord Record2("foo", 0x1234, {Max});
  auto Result2 = Writer.addRecord(std::move(Record2));
  ASSERT_EQ(Result2, instrprof_error::counter_overflow);

  InstrProfRecord Record3(bar, 0x9012, {8});
  auto Result3 = Writer.addRecord(std::move(Record3));
  ASSERT_EQ(Result3, instrprof_error::success);

  InstrProfRecord Record4("baz", 0x5678, {3, 4});
  Record4.reserveSites(IPVK_IndirectCallTarget, 1);
  InstrProfValueData VD4[] = {{uint64_t(bar), 1}};
  Record4.addValueData(IPVK_IndirectCallTarget, 0, VD4, 1, nullptr);
  auto Result4 = Writer.addRecord(std::move(Record4));
  ASSERT_EQ(Result4, instrprof_error::success);

  // Verify value data counter overflow.
  InstrProfRecord Record5("baz", 0x5678, {5, 6});
  Record5.reserveSites(IPVK_IndirectCallTarget, 1);
  InstrProfValueData VD5[] = {{uint64_t(bar), Max}};
  Record5.addValueData(IPVK_IndirectCallTarget, 0, VD5, 1, nullptr);
  auto Result5 = Writer.addRecord(std::move(Record5));
  ASSERT_EQ(Result5, instrprof_error::counter_overflow);

  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  // Verify saturation of counts.
  ErrorOr<InstrProfRecord> ReadRecord1 =
      Reader->getInstrProfRecord("foo", 0x1234);
  ASSERT_TRUE(NoError(ReadRecord1.getError()));
  ASSERT_EQ(Max, ReadRecord1.get().Counts[0]);

  ErrorOr<InstrProfRecord> ReadRecord2 =
      Reader->getInstrProfRecord("baz", 0x5678);
  ASSERT_EQ(1U, ReadRecord2.get().getNumValueSites(IPVK_IndirectCallTarget));
  std::unique_ptr<InstrProfValueData[]> VD =
      ReadRecord2.get().getValueForSite(IPVK_IndirectCallTarget, 0);
  ASSERT_EQ(StringRef("bar"), StringRef((const char *)VD[0].Value, 3));
  ASSERT_EQ(Max, VD[0].Count);
}

// This test tests that when there are too many values
// for a given site, the merged results are properly
// truncated.
TEST_F(InstrProfTest, get_icall_data_merge_site_trunc) {
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

  Writer.addRecord(std::move(Record11));
  // Merge profile data.
  Writer.addRecord(std::move(Record12));

  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  ErrorOr<InstrProfRecord> R = Reader->getInstrProfRecord("caller", 0x1234);
  ASSERT_TRUE(NoError(R.getError()));
  std::unique_ptr<InstrProfValueData[]> VD(
      R.get().getValueForSite(IPVK_IndirectCallTarget, 0));
  ASSERT_EQ(2U, R.get().getNumValueSites(IPVK_IndirectCallTarget));
  ASSERT_EQ(255U, R.get().getNumValueDataForSite(IPVK_IndirectCallTarget, 0));
  for (unsigned I = 0; I < 255; I++) {
    ASSERT_EQ(VD[I].Value, 509 - I);
    ASSERT_EQ(VD[I].Count, 1509 - I);
  }
}

// Synthesize runtime value profile data.
ValueProfNode Site1Values[5] = {{{uint64_t("callee1"), 400}, &Site1Values[1]},
                                {{uint64_t("callee2"), 1000}, &Site1Values[2]},
                                {{uint64_t("callee3"), 500}, &Site1Values[3]},
                                {{uint64_t("callee4"), 300}, &Site1Values[4]},
                                {{uint64_t("callee5"), 100}, 0}};

ValueProfNode Site2Values[4] = {{{uint64_t("callee5"), 800}, &Site2Values[1]},
                                {{uint64_t("callee3"), 1000}, &Site2Values[2]},
                                {{uint64_t("callee2"), 2500}, &Site2Values[3]},
                                {{uint64_t("callee1"), 1300}, 0}};

ValueProfNode Site3Values[3] = {{{uint64_t("callee6"), 800}, &Site3Values[1]},
                                {{uint64_t("callee3"), 1000}, &Site3Values[2]},
                                {{uint64_t("callee4"), 5500}, 0}};

ValueProfNode Site4Values[2] = {{{uint64_t("callee2"), 1800}, &Site4Values[1]},
                                {{uint64_t("callee3"), 2000}, 0}};

static ValueProfNode *ValueProfNodes[5] = {&Site1Values[0], &Site2Values[0],
                                           &Site3Values[0], &Site4Values[0], 0};
static uint16_t NumValueSites[IPVK_Last + 1] = {5};
TEST_F(InstrProfTest, runtime_value_prof_data_read_write) {
  ValueProfRuntimeRecord RTRecord;
  initializeValueProfRuntimeRecord(&RTRecord, &NumValueSites[0],
                                   &ValueProfNodes[0]);

  ValueProfData *VPData = serializeValueProfDataFromRT(&RTRecord, nullptr);

  InstrProfRecord Record("caller", 0x1234, {1ULL << 31, 2});

  VPData->deserializeTo(Record, 0);

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

  finalizeValueProfRuntimeRecord(&RTRecord);
  free(VPData);
}

TEST_F(InstrProfTest, get_max_function_count) {
  InstrProfRecord Record1("foo", 0x1234, {1ULL << 31, 2});
  InstrProfRecord Record2("bar", 0, {1ULL << 63});
  InstrProfRecord Record3("baz", 0x5678, {0, 0, 0, 0});
  Writer.addRecord(std::move(Record1));
  Writer.addRecord(std::move(Record2));
  Writer.addRecord(std::move(Record3));
  auto Profile = Writer.writeBuffer();
  readProfile(std::move(Profile));

  ASSERT_EQ(1ULL << 63, Reader->getMaximumFunctionCount());
}

TEST_F(InstrProfTest, get_weighted_function_counts) {
  InstrProfRecord Record1("foo", 0x1234, {1, 2});
  InstrProfRecord Record2("foo", 0x1235, {3, 4});
  Writer.addRecord(std::move(Record1), 3);
  Writer.addRecord(std::move(Record2), 5);
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

TEST_F(InstrProfTest, instr_prof_symtab_test) {
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

TEST_F(InstrProfTest, instr_prof_symtab_module_test) {
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
  ProfSymtab.create(*(M.get()));

  StringRef Funcs[] = {"Gfoo", "Gblah", "Gbar", "Ifoo", "Iblah", "Ibar",
                       "Pfoo", "Pblah", "Pbar", "Wfoo", "Wblah", "Wbar"};

  for (unsigned I = 0; I < sizeof(Funcs) / sizeof(*Funcs); I++) {
    Function *F = M->getFunction(Funcs[I]);
    ASSERT_TRUE(F != NULL);
    std::string PGOName = getPGOFuncName(*F);
    ASSERT_EQ(StringRef(PGOName),
              ProfSymtab.getFuncName(IndexedInstrProf::ComputeHash(PGOName)));
  }
}

TEST_F(InstrProfTest, instr_prof_symtab_compression_test) {
  std::vector<std::string> FuncNames1;
  std::vector<std::string> FuncNames2;
  for (int I = 0; I < 10 * 1024; I++) {
    std::string str;
    raw_string_ostream OS(str);
    OS << "func_" << I;
    FuncNames1.push_back(OS.str());
    str.clear();
    OS << "fooooooooooooooo_" << I;
    FuncNames1.push_back(OS.str());
    str.clear();
    OS << "BAR_" << I;
    FuncNames2.push_back(OS.str());
    str.clear();
    OS << "BlahblahBlahblahBar_" << I;
    FuncNames2.push_back(OS.str());
  }

  for (int Padding = 0; Padding < 10; Padding++) {
    for (int DoCompression = 0; DoCompression < 2; DoCompression++) {
      // Compressing:
      std::string FuncNameStrings1;
      collectPGOFuncNameStrings(FuncNames1,
                                (DoCompression != 0 && zlib::isAvailable()),
                                FuncNameStrings1);

      // Compressing:
      std::string FuncNameStrings2;
      collectPGOFuncNameStrings(FuncNames2,
                                (DoCompression != 0 && zlib::isAvailable()),
                                FuncNameStrings2);

      // Join with paddings:
      std::string FuncNameStrings = FuncNameStrings1;
      for (int P = 0; P < Padding; P++) {
        FuncNameStrings.push_back('\0');
      }
      FuncNameStrings += FuncNameStrings2;

      // Now decompress:
      InstrProfSymtab Symtab;
      Symtab.create(StringRef(FuncNameStrings));

      // Now do the checks:
      // First sampling some data points:
      StringRef R = Symtab.getFuncName(IndexedInstrProf::ComputeHash(FuncNames1[0]));
      ASSERT_EQ(StringRef("func_0"), R);
      R = Symtab.getFuncName(IndexedInstrProf::ComputeHash(FuncNames1[1]));
      ASSERT_EQ(StringRef("fooooooooooooooo_0"), R);
      R = Symtab.getFuncName(IndexedInstrProf::ComputeHash(FuncNames1[998]));
      ASSERT_EQ(StringRef("func_499"), R);
      R = Symtab.getFuncName(IndexedInstrProf::ComputeHash(FuncNames1[999]));
      ASSERT_EQ(StringRef("fooooooooooooooo_499"), R);
      R = Symtab.getFuncName(IndexedInstrProf::ComputeHash(FuncNames2[100]));
      ASSERT_EQ(StringRef("BAR_50"), R);
      R = Symtab.getFuncName(IndexedInstrProf::ComputeHash(FuncNames2[101]));
      ASSERT_EQ(StringRef("BlahblahBlahblahBar_50"), R);
      for (int I = 0; I < 10 * 1024; I++) {
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

} // end anonymous namespace
