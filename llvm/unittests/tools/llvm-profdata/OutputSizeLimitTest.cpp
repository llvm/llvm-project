//===- llvm/unittests/tools/llvm-profdata/OutputSizeLimitTest.cpp ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/SampleProfReader.h"
#include "llvm/ProfileData/SampleProfWriter.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using llvm::unittest::TempFile;

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

const char EmptyProfile[18] = "\xff\xe5\xd0\xb1\xf4\xc9\x94\xa8\x53\x67";

/// sys::fs and SampleProf mix Error and error_code, making an adapter class
/// to keep code elegant.
template <typename T> class ExpectedErrorOr : public Expected<T> {
public:
  ExpectedErrorOr(T &&Obj) : Expected<T>(Obj) {}

  ExpectedErrorOr(std::error_code EC) : Expected<T>(errorCodeToError(EC)) {}

  ExpectedErrorOr(Error &&E) : Expected<T>(std::move(E)) {}

  template <typename U>
  ExpectedErrorOr(ErrorOr<U> &&E)
      : Expected<T>(errorCodeToError(E.getError())) {}

  template <typename U>
  ExpectedErrorOr(Expected<U> &&E) : Expected<T>(E.takeError()) {}
};

#define DEF_VAR_RETURN_IF_ERROR(Var, Value)                                    \
  auto Var##OrErr = Value;                                                     \
  if (!Var##OrErr)                                                             \
    return Var##OrErr;                                                         \
  auto Var = std::move(Var##OrErr.get())

#define VAR_RETURN_IF_ERROR(Var, Value)                                        \
  Var##OrErr = Value;                                                          \
  if (!Var##OrErr)                                                             \
    return Var##OrErr;                                                         \
  Var = std::move(Var##OrErr.get())

#define RETURN_IF_ERROR(Value)                                                 \
  if (auto E = Value)                                                          \
  return std::move(E)

/// The main testing routine. After rewriting profiles with size limit, check
/// the following:
/// 1. The file size of the new profile is within the size limit.
/// 2. The new profile is a subset of the old profile, and the content of every
/// sample in the new profile is unchanged.
/// Note that even though by default samples with fewest total count are dropped
/// first, this is not a requirement. Samples can be dropped by any order.
static ExpectedErrorOr<void *> RunTest(StringRef Input, size_t SizeLimit,
                                       SampleProfileFormat Format,
                                       bool Compress = false) {
  // Read Input profile.
  auto FS = vfs::getRealFileSystem();
  LLVMContext Context;
  auto InputBuffer = MemoryBuffer::getMemBuffer(Input);
  DEF_VAR_RETURN_IF_ERROR(
      Reader, SampleProfileReader::create(InputBuffer, Context, *FS));
  RETURN_IF_ERROR(Reader->read());
  SampleProfileMap OldProfiles = Reader->getProfiles();

  // Rewrite it to a temp file with size limit.
  TempFile Temp("profile", "afdo", "", true);
  bool isEmpty = false;
  {
    DEF_VAR_RETURN_IF_ERROR(Writer,
                            SampleProfileWriter::create(Temp.path(), Format));
    if (Compress)
      Writer->setToCompressAllSections();
    std::error_code EC = Writer->writeWithSizeLimit(OldProfiles, SizeLimit);
    // too_large means no sample could be written because SizeLimit is too
    // small. Otherwise any other error code indicates unexpected failure.
    if (EC == sampleprof_error::too_large)
      isEmpty = true;
    else if (EC)
      return EC;
  }

  // Read the temp file to get new profiles. Use the default empty profile if
  // temp file was not written because size limit is too small.
  SampleProfileMap NewProfiles;
  InputBuffer = MemoryBuffer::getMemBuffer(StringRef(EmptyProfile, 17));
  DEF_VAR_RETURN_IF_ERROR(
      NewReader, SampleProfileReader::create(InputBuffer, Context, *FS));
  if (!isEmpty) {
    VAR_RETURN_IF_ERROR(NewReader, SampleProfileReader::create(
                                       Temp.path().str(), Context, *FS));
    RETURN_IF_ERROR(NewReader->read());
    NewProfiles = NewReader->getProfiles();
  }

  // Check temp file is actually within size limit.
  uint64_t FileSize;
  RETURN_IF_ERROR(sys::fs::file_size(Temp.path(), FileSize));
  EXPECT_LE(FileSize, SizeLimit);

  // For every sample in the new profile, confirm it is in the old profile and
  // unchanged.
  for (auto Sample : NewProfiles) {
    auto FindResult = OldProfiles.find(Sample.first);
    EXPECT_NE(FindResult, OldProfiles.end());
    if (FindResult != OldProfiles.end()) {
      EXPECT_EQ(Sample.second.getHeadSamples(),
                FindResult->second.getHeadSamples());
      EXPECT_EQ(Sample.second, FindResult->second);
    }
  }
  return nullptr;
}

TEST(TestOutputSizeLimit, TestOutputSizeLimitExtBinary) {
  for (size_t OutputSizeLimit : {490, 489, 488, 475, 474, 459, 400})
    ASSERT_THAT_EXPECTED(
        RunTest(Input1, OutputSizeLimit, llvm::sampleprof::SPF_Ext_Binary),
        Succeeded());
}

TEST(TestOutputSizeLimit, TestOutputSizeLimitBinary) {
  for (size_t OutputSizeLimit : {250, 249, 248, 237, 236, 223, 200})
    ASSERT_THAT_EXPECTED(
        RunTest(Input1, OutputSizeLimit, llvm::sampleprof::SPF_Binary),
        Succeeded());
}

TEST(TestOutputSizeLimit, TestOutputSizeLimitText) {
  for (size_t OutputSizeLimit :
       {229, 228, 227, 213, 212, 211, 189, 188, 187, 186, 150})
    ASSERT_THAT_EXPECTED(
        RunTest(Input1, OutputSizeLimit, llvm::sampleprof::SPF_Text),
        Succeeded());
}

#if LLVM_ENABLE_ZLIB
TEST(TestOutputSizeLimit, TestOutputSizeLimitExtBinaryCompressed) {
  for (size_t OutputSizeLimit :
       {507, 506, 505, 494, 493, 492, 483, 482, 481, 480})
    ASSERT_THAT_EXPECTED(RunTest(Input1, OutputSizeLimit,
                                 llvm::sampleprof::SPF_Ext_Binary, true),
                         Succeeded());
}
#endif
