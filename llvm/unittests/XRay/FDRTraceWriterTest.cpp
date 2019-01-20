//===- llvm/unittest/XRay/FDRTraceWriterTest.cpp ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test a utility that can write out XRay FDR Mode formatted trace files.
//
//===----------------------------------------------------------------------===//
#include "llvm/XRay/FDRTraceWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/XRay/FDRLogBuilder.h"
#include "llvm/XRay/FDRRecords.h"
#include "llvm/XRay/Trace.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <string>

namespace llvm {
namespace xray {
namespace {

using testing::ElementsAre;
using testing::Eq;
using testing::Field;
using testing::IsEmpty;
using testing::Not;

// We want to be able to create an instance of an FDRTraceWriter and associate
// it with a stream, which could be loaded and turned into a Trace instance.
// This test writes out version 3 trace logs.
TEST(FDRTraceWriterTest, WriteToStringBufferVersion3) {
  std::string Data;
  raw_string_ostream OS(Data);
  XRayFileHeader H;
  H.Version = 3;
  H.Type = 1;
  H.ConstantTSC = true;
  H.NonstopTSC = true;
  H.CycleFrequency = 3e9;
  FDRTraceWriter Writer(OS, H);
  auto L = LogBuilder()
               .add<BufferExtents>(80)
               .add<NewBufferRecord>(1)
               .add<WallclockRecord>(1, 1)
               .add<PIDRecord>(1)
               .add<NewCPUIDRecord>(1, 2)
               .add<FunctionRecord>(RecordTypes::ENTER, 1, 1)
               .add<FunctionRecord>(RecordTypes::EXIT, 1, 100)
               .consume();
  for (auto &P : L)
    ASSERT_FALSE(errorToBool(P->apply(Writer)));
  OS.flush();

  // Then from here we load the Trace file.
  DataExtractor DE(Data, sys::IsLittleEndianHost, 8);
  auto TraceOrErr = loadTrace(DE, true);
  if (!TraceOrErr)
    FAIL() << TraceOrErr.takeError();
  auto &Trace = TraceOrErr.get();

  ASSERT_THAT(Trace, Not(IsEmpty()));
  EXPECT_THAT(Trace, ElementsAre(Field(&XRayRecord::FuncId, Eq(1)),
                                 Field(&XRayRecord::FuncId, Eq(1))));
  EXPECT_THAT(Trace, ElementsAre(Field(&XRayRecord::TId, Eq(1u)),
                                 Field(&XRayRecord::TId, Eq(1u))));
  EXPECT_THAT(Trace, ElementsAre(Field(&XRayRecord::PId, Eq(1u)),
                                 Field(&XRayRecord::PId, Eq(1u))));
  EXPECT_THAT(Trace, ElementsAre(Field(&XRayRecord::CPU, Eq(1u)),
                                 Field(&XRayRecord::CPU, Eq(1u))));
  EXPECT_THAT(Trace,
              ElementsAre(Field(&XRayRecord::Type, Eq(RecordTypes::ENTER)),
                          Field(&XRayRecord::Type, Eq(RecordTypes::EXIT))));
}

// This version is almost exactly the same as above, except writing version 2
// logs, without the PID records.
TEST(FDRTraceWriterTest, WriteToStringBufferVersion2) {
  std::string Data;
  raw_string_ostream OS(Data);
  XRayFileHeader H;
  H.Version = 2;
  H.Type = 1;
  H.ConstantTSC = true;
  H.NonstopTSC = true;
  H.CycleFrequency = 3e9;
  FDRTraceWriter Writer(OS, H);
  auto L = LogBuilder()
               .add<BufferExtents>(64)
               .add<NewBufferRecord>(1)
               .add<WallclockRecord>(1, 1)
               .add<NewCPUIDRecord>(1, 2)
               .add<FunctionRecord>(RecordTypes::ENTER, 1, 1)
               .add<FunctionRecord>(RecordTypes::EXIT, 1, 100)
               .consume();
  for (auto &P : L)
    ASSERT_FALSE(errorToBool(P->apply(Writer)));
  OS.flush();

  // Then from here we load the Trace file.
  DataExtractor DE(Data, sys::IsLittleEndianHost, 8);
  auto TraceOrErr = loadTrace(DE, true);
  if (!TraceOrErr)
    FAIL() << TraceOrErr.takeError();
  auto &Trace = TraceOrErr.get();

  ASSERT_THAT(Trace, Not(IsEmpty()));
  EXPECT_THAT(Trace, ElementsAre(Field(&XRayRecord::FuncId, Eq(1)),
                                 Field(&XRayRecord::FuncId, Eq(1))));
  EXPECT_THAT(Trace, ElementsAre(Field(&XRayRecord::TId, Eq(1u)),
                                 Field(&XRayRecord::TId, Eq(1u))));
  EXPECT_THAT(Trace, ElementsAre(Field(&XRayRecord::CPU, Eq(1u)),
                                 Field(&XRayRecord::CPU, Eq(1u))));
  EXPECT_THAT(Trace,
              ElementsAre(Field(&XRayRecord::Type, Eq(RecordTypes::ENTER)),
                          Field(&XRayRecord::Type, Eq(RecordTypes::EXIT))));
}

// This covers version 1 of the log, without a BufferExtents record but has an
// explicit EndOfBuffer record.
TEST(FDRTraceWriterTest, WriteToStringBufferVersion1) {
  std::string Data;
  raw_string_ostream OS(Data);
  XRayFileHeader H;
  H.Version = 1;
  H.Type = 1;
  H.ConstantTSC = true;
  H.NonstopTSC = true;
  H.CycleFrequency = 3e9;
  // Write the size of buffers out, arbitrarily it's 4k.
  constexpr uint64_t BufferSize = 4096;
  std::memcpy(H.FreeFormData, reinterpret_cast<const char *>(&BufferSize),
              sizeof(BufferSize));
  FDRTraceWriter Writer(OS, H);
  OS.flush();

  // Ensure that at this point the Data buffer has the file header serialized
  // size.
  ASSERT_THAT(Data.size(), Eq(32u));
  auto L = LogBuilder()
               .add<NewBufferRecord>(1)
               .add<WallclockRecord>(1, 1)
               .add<NewCPUIDRecord>(1, 2)
               .add<FunctionRecord>(RecordTypes::ENTER, 1, 1)
               .add<FunctionRecord>(RecordTypes::EXIT, 1, 100)
               .add<EndBufferRecord>()
               .consume();
  for (auto &P : L)
    ASSERT_FALSE(errorToBool(P->apply(Writer)));

  // We need to pad the buffer with 4016 (4096 - 80) bytes of zeros.
  OS.write_zeros(4016);
  OS.flush();

  // For version 1 of the log, we need the whole buffer to be the size of the
  // file header plus 32.
  ASSERT_THAT(Data.size(), Eq(BufferSize + 32));

  // Then from here we load the Trace file.
  DataExtractor DE(Data, sys::IsLittleEndianHost, 8);
  auto TraceOrErr = loadTrace(DE, true);
  if (!TraceOrErr)
    FAIL() << TraceOrErr.takeError();
  auto &Trace = TraceOrErr.get();

  ASSERT_THAT(Trace, Not(IsEmpty()));
  EXPECT_THAT(Trace, ElementsAre(Field(&XRayRecord::FuncId, Eq(1)),
                                 Field(&XRayRecord::FuncId, Eq(1))));
  EXPECT_THAT(Trace, ElementsAre(Field(&XRayRecord::TId, Eq(1u)),
                                 Field(&XRayRecord::TId, Eq(1u))));
  EXPECT_THAT(Trace, ElementsAre(Field(&XRayRecord::CPU, Eq(1u)),
                                 Field(&XRayRecord::CPU, Eq(1u))));
  EXPECT_THAT(Trace,
              ElementsAre(Field(&XRayRecord::Type, Eq(RecordTypes::ENTER)),
                          Field(&XRayRecord::Type, Eq(RecordTypes::EXIT))));
}

} // namespace
} // namespace xray
} // namespace llvm
