//===- llvm/unittest/XRay/FDRProducerConsumerTest.cpp -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include <tuple>

namespace llvm {
namespace xray {
namespace {

using ::testing::Eq;
using ::testing::IsEmpty;
using ::testing::Not;

template <class RecordType> std::unique_ptr<Record> MakeRecord();

template <> std::unique_ptr<Record> MakeRecord<BufferExtents>() {
  return make_unique<BufferExtents>(1);
}

template <> std::unique_ptr<Record> MakeRecord<NewBufferRecord>() {
  return make_unique<NewBufferRecord>(1);
}

template <> std::unique_ptr<Record> MakeRecord<NewCPUIDRecord>() {
  return make_unique<NewCPUIDRecord>(1, 2);
}

template <> std::unique_ptr<Record> MakeRecord<TSCWrapRecord>() {
  return make_unique<TSCWrapRecord>(1);
}

template <> std::unique_ptr<Record> MakeRecord<WallclockRecord>() {
  return make_unique<WallclockRecord>(1, 2);
}

template <> std::unique_ptr<Record> MakeRecord<CustomEventRecord>() {
  return make_unique<CustomEventRecord>(4, 1, "data");
}

template <> std::unique_ptr<Record> MakeRecord<CallArgRecord>() {
  return make_unique<CallArgRecord>(1);
}

template <> std::unique_ptr<Record> MakeRecord<PIDRecord>() {
  return make_unique<PIDRecord>(1);
}

template <> std::unique_ptr<Record> MakeRecord<FunctionRecord>() {
  return make_unique<FunctionRecord>(RecordTypes::ENTER, 1, 2);
}

template <class T> class RoundTripTest : public ::testing::Test {
public:
  RoundTripTest() : Data(), OS(Data) {
    H.Version = 3;
    H.Type = 1;
    H.ConstantTSC = true;
    H.NonstopTSC = true;
    H.CycleFrequency = 3e9;

    Writer = make_unique<FDRTraceWriter>(OS, H);
    Rec = MakeRecord<T>();
  }

protected:
  std::string Data;
  raw_string_ostream OS;
  XRayFileHeader H;
  std::unique_ptr<FDRTraceWriter> Writer;
  std::unique_ptr<Record> Rec;
};

TYPED_TEST_CASE_P(RoundTripTest);

// This test ensures that the writing and reading implementations are in sync --
// that given write(read(write(R))) == R.
TYPED_TEST_P(RoundTripTest, RoundTripsSingleValue) {
  auto &R = this->Rec;
  ASSERT_FALSE(errorToBool(R->apply(*this->Writer)));
  this->OS.flush();

  DataExtractor DE(this->Data, sys::IsLittleEndianHost, 8);
  uint32_t OffsetPtr = 0;
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
  OS2.flush();

  EXPECT_EQ(Data2.substr(sizeof(XRayFileHeader)),
            this->Data.substr(sizeof(XRayFileHeader)));
  EXPECT_THAT(Records[0]->type(), Eq(R->type()));
}

REGISTER_TYPED_TEST_CASE_P(RoundTripTest, RoundTripsSingleValue);

using RecordTypes =
    ::testing::Types<BufferExtents, NewBufferRecord, NewCPUIDRecord,
                     TSCWrapRecord, WallclockRecord, CustomEventRecord,
                     CallArgRecord, BufferExtents, PIDRecord, FunctionRecord>;
INSTANTIATE_TYPED_TEST_CASE_P(Records, RoundTripTest, RecordTypes);

} // namespace
} // namespace xray
} // namespace llvm
