//===- llvm/unittest/XRay/FDRProducerConsumerTest.cpp -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test for round-trip record writing and reading.
//
//===----------------------------------------------------------------------===//
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/XRay/FDRLogBuilder.h"
#include "llvm/XRay/FDRRecordConsumer.h"
#include "llvm/XRay/FDRRecordProducer.h"
#include "llvm/XRay/FDRRecords.h"
#include "llvm/XRay/FDRTraceWriter.h"
#include "llvm/XRay/FileHeaderReader.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <string>

namespace llvm {
namespace xray {
namespace {

using ::testing::Eq;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::SizeIs;

template <class RecordType> std::unique_ptr<Record> MakeRecord();

template <> std::unique_ptr<Record> MakeRecord<NewBufferRecord>() {
  return std::make_unique<NewBufferRecord>(1);
}

template <> std::unique_ptr<Record> MakeRecord<NewCPUIDRecord>() {
  return std::make_unique<NewCPUIDRecord>(1, 2);
}

template <> std::unique_ptr<Record> MakeRecord<TSCWrapRecord>() {
  return std::make_unique<TSCWrapRecord>(1);
}

template <> std::unique_ptr<Record> MakeRecord<WallclockRecord>() {
  return std::make_unique<WallclockRecord>(1, 2);
}

template <> std::unique_ptr<Record> MakeRecord<CustomEventRecord>() {
  return std::make_unique<CustomEventRecord>(4, 1, 2, "data");
}

template <> std::unique_ptr<Record> MakeRecord<CallArgRecord>() {
  return std::make_unique<CallArgRecord>(1);
}

template <> std::unique_ptr<Record> MakeRecord<PIDRecord>() {
  return std::make_unique<PIDRecord>(1);
}

template <> std::unique_ptr<Record> MakeRecord<FunctionRecord>() {
  return std::make_unique<FunctionRecord>(RecordTypes::ENTER, 1, 2);
}

template <> std::unique_ptr<Record> MakeRecord<CustomEventRecordV5>() {
  return std::make_unique<CustomEventRecordV5>(4, 1, "data");
}

template <> std::unique_ptr<Record> MakeRecord<TypedEventRecord>() {
  return std::make_unique<TypedEventRecord>(4, 1, 2, "data");
}

template <class T> class RoundTripTest : public ::testing::Test {
public:
  RoundTripTest() : Data(), OS(Data) {
    H.Version = 4;
    H.Type = 1;
    H.ConstantTSC = true;
    H.NonstopTSC = true;
    H.CycleFrequency = 3e9;

    Writer = std::make_unique<FDRTraceWriter>(OS, H);
    Rec = MakeRecord<T>();
  }

protected:
  std::string Data;
  raw_string_ostream OS;
  XRayFileHeader H;
  std::unique_ptr<FDRTraceWriter> Writer;
  std::unique_ptr<Record> Rec;
};

TYPED_TEST_SUITE_P(RoundTripTest);

template <class T> class RoundTripTestV5 : public ::testing::Test {
public:
  RoundTripTestV5() : Data(), OS(Data) {
    H.Version = 5;
    H.Type = 1;
    H.ConstantTSC = true;
    H.NonstopTSC = true;
    H.CycleFrequency = 3e9;

    Writer = std::make_unique<FDRTraceWriter>(OS, H);
    Rec = MakeRecord<T>();
  }

protected:
  std::string Data;
  raw_string_ostream OS;
  XRayFileHeader H;
  std::unique_ptr<FDRTraceWriter> Writer;
  std::unique_ptr<Record> Rec;
};

TYPED_TEST_SUITE_P(RoundTripTestV5);

// This test ensures that the writing and reading implementations are in sync --
// that given write(read(write(R))) == R.
TYPED_TEST_P(RoundTripTest, RoundTripsSingleValue) {
  // Always write a buffer extents record which will cover the correct size of
  // the record, for version 3 and up.
  BufferExtents BE(200);
  ASSERT_FALSE(errorToBool(BE.apply(*this->Writer)));
  auto &R = this->Rec;
  ASSERT_FALSE(errorToBool(R->apply(*this->Writer)));

  DataExtractor DE(this->Data, sys::IsLittleEndianHost, 8);
  uint64_t OffsetPtr = 0;
  auto HeaderOrErr = readBinaryFormatHeader(DE, OffsetPtr);
  if (!HeaderOrErr)
    FAIL() << HeaderOrErr.takeError();

  FileBasedRecordProducer P(HeaderOrErr.get(), DE, OffsetPtr);
  std::vector<std::unique_ptr<Record>> Records;
  LogBuilderConsumer C(Records);
  while (DE.isValidOffsetForDataOfSize(OffsetPtr, 1)) {
    auto R = P.produce();
    if (!R)
      FAIL() << R.takeError();
    if (auto E = C.consume(std::move(R.get())))
      FAIL() << E;
  }
  ASSERT_THAT(Records, Not(IsEmpty()));
  std::string Data2;
  raw_string_ostream OS2(Data2);
  FDRTraceWriter Writer2(OS2, this->H);
  for (auto &P : Records)
    ASSERT_FALSE(errorToBool(P->apply(Writer2)));

  EXPECT_EQ(Data2.substr(sizeof(XRayFileHeader)),
            this->Data.substr(sizeof(XRayFileHeader)));
  ASSERT_THAT(Records, SizeIs(2));
  EXPECT_THAT(Records[1]->getRecordType(), Eq(R->getRecordType()));
}

REGISTER_TYPED_TEST_SUITE_P(RoundTripTest, RoundTripsSingleValue);

// We duplicate the above case for the V5 version using different types and
// encodings.
TYPED_TEST_P(RoundTripTestV5, RoundTripsSingleValue) {
  BufferExtents BE(200);
  ASSERT_FALSE(errorToBool(BE.apply(*this->Writer)));
  auto &R = this->Rec;
  ASSERT_FALSE(errorToBool(R->apply(*this->Writer)));

  DataExtractor DE(this->Data, sys::IsLittleEndianHost, 8);
  uint64_t OffsetPtr = 0;
  auto HeaderOrErr = readBinaryFormatHeader(DE, OffsetPtr);
  if (!HeaderOrErr)
    FAIL() << HeaderOrErr.takeError();

  FileBasedRecordProducer P(HeaderOrErr.get(), DE, OffsetPtr);
  std::vector<std::unique_ptr<Record>> Records;
  LogBuilderConsumer C(Records);
  while (DE.isValidOffsetForDataOfSize(OffsetPtr, 1)) {
    auto R = P.produce();
    if (!R)
      FAIL() << R.takeError();
    if (auto E = C.consume(std::move(R.get())))
      FAIL() << E;
  }
  ASSERT_THAT(Records, Not(IsEmpty()));
  std::string Data2;
  raw_string_ostream OS2(Data2);
  FDRTraceWriter Writer2(OS2, this->H);
  for (auto &P : Records)
    ASSERT_FALSE(errorToBool(P->apply(Writer2)));

  EXPECT_EQ(Data2.substr(sizeof(XRayFileHeader)),
            this->Data.substr(sizeof(XRayFileHeader)));
  ASSERT_THAT(Records, SizeIs(2));
  EXPECT_THAT(Records[1]->getRecordType(), Eq(R->getRecordType()));
}

REGISTER_TYPED_TEST_SUITE_P(RoundTripTestV5, RoundTripsSingleValue);

// These are the record types we support for v4 and below.
using RecordTypes =
    ::testing::Types<NewBufferRecord, NewCPUIDRecord, TSCWrapRecord,
                     WallclockRecord, CustomEventRecord, CallArgRecord,
                     PIDRecord, FunctionRecord>;
INSTANTIATE_TYPED_TEST_SUITE_P(Records, RoundTripTest, RecordTypes, );

// For V5, we have two new types we're supporting.
using RecordTypesV5 =
    ::testing::Types<NewBufferRecord, NewCPUIDRecord, TSCWrapRecord,
                     WallclockRecord, CustomEventRecordV5, TypedEventRecord,
                     CallArgRecord, PIDRecord, FunctionRecord>;
INSTANTIATE_TYPED_TEST_SUITE_P(Records, RoundTripTestV5, RecordTypesV5, );

} // namespace
} // namespace xray
} // namespace llvm
