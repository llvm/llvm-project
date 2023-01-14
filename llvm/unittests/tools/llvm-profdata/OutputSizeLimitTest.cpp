//===- llvm/unittests/tools/llvm-profdata/OutputSizeLimitTest.cpp ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Correctness check of SampleProfileWriter::writeWithSizeLimit() is done in
// llvm lit tests (llvm/test/tools/llvm-profdata/output-size-limit.test). This
// test checks the output size is in fact under the specified size limit.

#include "llvm/ProfileData/SampleProfReader.h"
#include "llvm/ProfileData/SampleProfWriter.h"
#include "gtest/gtest.h"

using namespace llvm;

std::string Input1 = R"(main:184019:0
 4: 534
 4.2: 534
 5: 1075
 5.1: 1075
 6: 2080
 7: 534
 9: 2064 _Z3bari:1471 _Z3fooi:631
 10: inline1:1000
  1: 1000
 10: inline2:2000
  1: 2000
_Z3bari:20301:1437
 1: 1437
_Z3fooi:7711:610
 1: 610)";

static void CheckAssertFalse(std::error_code t) {
  ASSERT_FALSE(t) << t.message().c_str();
}

static SampleProfileMap ReadInput(StringRef Input) {
  LLVMContext Context;
  auto InputBuffer = MemoryBuffer::getMemBufferCopy(Input);
  auto ReaderOrErr = SampleProfileReader::create(InputBuffer, Context);
  CheckAssertFalse(ReaderOrErr.getError());
  auto Reader = std::move(ReaderOrErr.get());
  CheckAssertFalse(Reader->read());
  return Reader->getProfiles();
}

static std::unique_ptr<SampleProfileWriter>
CreateWriter(SmallVector<char> &OutputBuffer) {
  std::unique_ptr<raw_ostream> BufferStream(
      new raw_svector_ostream(OutputBuffer));
  auto WriterOrErr = SampleProfileWriter::create(
      BufferStream, llvm::sampleprof::SPF_Ext_Binary);
  CheckAssertFalse(WriterOrErr.getError());
  return std::move(WriterOrErr.get());
}

// Returns the actual size of the written profile.
static size_t WriteProfile(StringRef Input, size_t SizeLimit) {
  SampleProfileMap Profiles = ReadInput(Input);
  SmallVector<char> OutputBuffer;
  auto Writer = CreateWriter(OutputBuffer);
  std::error_code EC = Writer->writeWithSizeLimit(Profiles, SizeLimit);
  // too_large means no sample could be written because SizeLimit is too small.
  // Otherwise any other error code indicates unexpected failure.
  if (EC == sampleprof_error::too_large)
    return 0;
  else
    CheckAssertFalse(EC);
  return OutputBuffer.size();
}

TEST(TestOutputSizeLimit, TestOutputSizeLimit1) {
  for (size_t OutputSizeLimit : {489, 488, 474, 400})
    EXPECT_LE(WriteProfile(Input1, OutputSizeLimit), OutputSizeLimit);
}
