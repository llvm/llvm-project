//===- PerfJITUnwindInfoTest.cpp - Test perf CFI -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PerfJITUnwindInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <algorithm>

#ifdef __linux__
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ExecutionEngine/Orc/Shared/PerfSharedStructs.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderPerf.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#endif

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::dwarf;

namespace {

// Model a compiler's input allocation separately from perf's synthetic ELF.
// Decode output with LLVM's independent DWARFDebugFrame parser.
struct InputCFI {
  std::string Bytes;
  uint8_t Version = 1, Encoding = 0x1b;
  uint64_t FrameAddr = 0x10000000, CodeAddr = 0x10000800, CodeSize = 64;
  std::string Augmentation = "zR", FDEProgram;

  static void u32(std::string &Out, uint32_t Value) {
    for (unsigned I = 0; I != 4; ++I)
      Out += char(Value >> (8 * I));
  }
  uint64_t record(std::string Body) {
    uint64_t Offset = Bytes.size();
    while ((Body.size() + 4) % 8)
      Body += '\0';
    u32(Bytes, Body.size());
    Bytes += Body;
    return Offset;
  }
  uint64_t cie() {
    std::string Body(4, '\0');
    Body += char(Version);
    Body += Augmentation;
    Body += '\0';
    Body.append("\x01\x78\x10",
                3); // Code/data alignment 1/-8, return register 16.
    if (Augmentation == "zPLR") {
      Body += char(7);
      Body +=
          char(0x9b); // Indirect PC-relative personality; never dereferenced.
      u32(Body, 0x1234);
      Body += char(0x1b); // LSDA encoding.
    } else {
      Body += char(1);
    }
    Body += char(Encoding);
    Body.append("\x0c\x07\x08\x90\x01",
                5); // CFA rsp+8; return address at CFA-8.
    return record(std::move(Body));
  }
  void fde(uint64_t CIEOffset) {
    uint64_t Offset = Bytes.size();
    std::string Body;
    u32(Body, Offset + 4 - CIEOffset);
    u32(Body, CodeAddr - (FrameAddr + Offset + 8));
    u32(Body, CodeSize);
    Body += char(Augmentation == "zPLR" ? 4 : 0);
    if (Augmentation == "zPLR")
      u32(Body, 0x5678); // LSDA pointer; omitted in the profiler copy.
    Body += FDEProgram;
    record(std::move(Body));
  }
  void finish() { u32(Bytes, 0); }
  void make() {
    fde(cie());
    finish();
  }
};

void checkOutput(StringRef Output, uint64_t CodeSize,
                 ArrayRef<std::pair<uint64_t, uint64_t>> Expected,
                 unsigned ExpectedCIEs = 1) {
  DWARFDebugFrame Parsed(Triple::x86_64, true,
                         0x80 + alignTo(CodeSize, uint64_t(8)));
  ASSERT_THAT_ERROR(Parsed.parse(DWARFDataExtractor(Output, true, 8)),
                    Succeeded());
  SmallVector<std::pair<uint64_t, uint64_t>> Locations;
  unsigned CIEs = 0;
  for (const auto &Entry : Parsed.entries()) {
    if (!Entry.getLength())
      continue;
    if (auto *C = dyn_cast<CIE>(&Entry)) {
      ++CIEs;
      EXPECT_EQ(C->getAugmentationString(), "zR");
      EXPECT_EQ(C->getFDEPointerEncoding(), 0x1bu);
      EXPECT_FALSE(C->getPersonalityAddress());
      EXPECT_EQ(C->getLSDAPointerEncoding(), DW_EH_PE_omit);
    } else {
      auto &F = cast<FDE>(Entry);
      Locations.emplace_back(F.getInitialLocation(), F.getAddressRange());
      EXPECT_FALSE(F.getLSDAAddress());
    }
  }
  EXPECT_THAT(Locations, ::testing::ElementsAreArray(Expected));
  EXPECT_EQ(CIEs, ExpectedCIEs);
}

TEST(PerfJITUnwindInfoTest, CommonCompilerCFI) {
  for (StringRef Augmentation : {"zR", "zPLR"})
    for (uint8_t Version : {1, 3})
      for (uint64_t Size : {1, 7, 8, 9, 64}) {
        InputCFI I;
        I.Augmentation = Augmentation.str();
        I.Version = Version;
        I.CodeSize = Size;
        I.make();
        std::string Before = I.Bytes;
        auto Info = PerfJITUnwindInfo::create(I.Bytes, I.FrameAddr, 8, true);
        ASSERT_THAT_EXPECTED(Info, Succeeded());
        auto Output = Info->emitEHFrame(I.CodeAddr, I.CodeSize);
        ASSERT_THAT_EXPECTED(Output, Succeeded());
        checkOutput(*Output, I.CodeSize, {{0x80, I.CodeSize}});
        auto Again = Info->emitEHFrame(I.CodeAddr, I.CodeSize);
        ASSERT_THAT_EXPECTED(Again, Succeeded());
        EXPECT_EQ(*Again, *Output);
        EXPECT_EQ(I.Bytes, Before);
      }
}

TEST(PerfJITUnwindInfoTest, MultipleFDEsAndCIEs) {
  for (bool SeparateCIEs : {false, true}) {
    InputCFI I;
    I.CodeSize = 16;
    uint64_t PC = I.CodeAddr;
    auto C = I.cie();
    I.fde(C);
    I.CodeAddr += 16;
    if (SeparateCIEs) {
      I.Augmentation = "zPLR";
      C = I.cie();
    }
    I.fde(C);
    I.finish();
    auto Info = PerfJITUnwindInfo::create(I.Bytes, I.FrameAddr, 8, true);
    ASSERT_THAT_EXPECTED(Info, Succeeded());
    auto Output = Info->emitEHFrame(PC, 32);
    ASSERT_THAT_EXPECTED(Output, Succeeded());
    checkOutput(*Output, 32, {{0x80, 16}, {0x90, 16}}, SeparateCIEs ? 2 : 1);
    Output = Info->emitEHFrame(I.CodeAddr, 16);
    ASSERT_THAT_EXPECTED(Output, Succeeded());
    checkOutput(*Output, 16, {{0x80, 16}});
  }
}

TEST(PerfJITUnwindInfoTest, SharedFDEAndEmptyRanges) {
  InputCFI I;
  I.CodeSize = 512;
  I.make();
  auto Info = PerfJITUnwindInfo::create(I.Bytes, I.FrameAddr, 8, true);
  ASSERT_THAT_EXPECTED(Info, Succeeded());
  auto Output = Info->emitEHFrame(I.CodeAddr + 8, 31);
  ASSERT_THAT_EXPECTED(Output, Succeeded());
  checkOutput(*Output, 31, {{0x78, 512}});
  EXPECT_THAT_EXPECTED(Info->emitEHFrame(I.CodeAddr + 129, 16), Failed());
  EXPECT_THAT_EXPECTED(Info->emitEHFrame(I.CodeAddr + 1, 0), Failed());
  EXPECT_THAT_EXPECTED(Info->emitEHFrame(I.CodeAddr + 512, 1), Failed());
}

TEST(PerfJITUnwindInfoTest, SignedDisplacementAndHighAddresses) {
  for (uint64_t PC : {uint64_t(0x0ffff000), UINT64_MAX - 15}) {
    InputCFI I;
    I.CodeAddr = PC;
    I.CodeSize = 8;
    if (PC > INT64_MAX)
      I.FrameAddr = PC - 0x1000;
    I.make();
    auto Info = PerfJITUnwindInfo::create(I.Bytes, I.FrameAddr, 8, true);
    ASSERT_THAT_EXPECTED(Info, Succeeded());
    auto Output = Info->emitEHFrame(I.CodeAddr, I.CodeSize);
    ASSERT_THAT_EXPECTED(Output, Succeeded());
    checkOutput(*Output, I.CodeSize, {{0x80, I.CodeSize}});
  }
}

TEST(PerfJITUnwindInfoTest, AddressOverflowIsDeclined) {
  for (bool Underflow : {false, true}) {
    InputCFI I;
    I.FrameAddr = Underflow ? 0x20 : UINT64_MAX - 0x100;
    I.CodeAddr = Underflow ? UINT64_MAX - 0x100 : 0x100;
    I.make();
    EXPECT_THAT_EXPECTED(
        PerfJITUnwindInfo::create(I.Bytes, I.FrameAddr, 8, true), Failed());
  }
  InputCFI I;
  I.make();
  EXPECT_THAT_EXPECTED(PerfJITUnwindInfo::create(I.Bytes, UINT64_MAX, 8, true),
                       Failed());
  auto Info = PerfJITUnwindInfo::create(I.Bytes, I.FrameAddr, 8, true);
  ASSERT_THAT_EXPECTED(Info, Succeeded());
  EXPECT_THAT_EXPECTED(Info->emitEHFrame(UINT64_MAX - 1, 8), Failed());
  EXPECT_THAT_EXPECTED(Info->emitEHFrame(I.CodeAddr, UINT64_MAX), Failed());
}

TEST(PerfJITUnwindInfoTest, OutputSignedOffsetLimit) {
  InputCFI I;
  I.make();
  auto Info = PerfJITUnwindInfo::create(I.Bytes, I.FrameAddr, 8, true);
  ASSERT_THAT_EXPECTED(Info, Succeeded());
  auto Ordinary = Info->emitEHFrame(I.CodeAddr, I.CodeSize);
  ASSERT_THAT_EXPECTED(Ordinary, Succeeded());
  DWARFDebugFrame Parsed(Triple::x86_64, true, 0x80 + I.CodeSize);
  ASSERT_THAT_ERROR(Parsed.parse(DWARFDataExtractor(*Ordinary, true, 8)),
                    Succeeded());
  for (const auto &Entry : Parsed.entries())
    if (isa<FDE>(Entry)) {
      uint64_t Size = (uint64_t(1) << 31) - (Entry.getOffset() + 8);
      auto Output = Info->emitEHFrame(I.CodeAddr, Size);
      ASSERT_THAT_EXPECTED(Output, Succeeded());
      checkOutput(*Output, Size, {{0x80, I.CodeSize}});
      EXPECT_THAT_EXPECTED(Info->emitEHFrame(I.CodeAddr, Size + 8), Failed());
    }
}

TEST(PerfJITUnwindInfoTest, UnsupportedFormatsAreDeclined) {
  InputCFI I;
  I.make();
  EXPECT_THAT_EXPECTED(PerfJITUnwindInfo::create(I.Bytes, I.FrameAddr, 4, true),
                       Failed());
  EXPECT_THAT_EXPECTED(
      PerfJITUnwindInfo::create(I.Bytes, I.FrameAddr, 8, false), Failed());
  std::string DW64(4, char(0xff));
  EXPECT_THAT_EXPECTED(PerfJITUnwindInfo::create(DW64, I.FrameAddr, 8, true),
                       Failed());
  for (StringRef Augmentation : {"zRS", "zRX", "zRPL"}) {
    InputCFI Other;
    Other.Augmentation = Augmentation.str();
    Other.make();
    EXPECT_THAT_EXPECTED(
        PerfJITUnwindInfo::create(Other.Bytes, Other.FrameAddr, 8, true),
        Failed());
  }
  I.Bytes.clear();
  I.Encoding = 0x9b;
  I.make();
  EXPECT_THAT_EXPECTED(PerfJITUnwindInfo::create(I.Bytes, I.FrameAddr, 8, true),
                       Failed());
}

TEST(PerfJITUnwindInfoTest, UnsupportedProgramOnlyAffectsSelectedFunction) {
  const uint8_t Opcodes[] = {DW_CFA_set_loc, DW_CFA_def_cfa_expression,
                             DW_CFA_expression, DW_CFA_val_expression, 0x17};
  for (uint8_t Op : Opcodes) {
    InputCFI I;
    auto C = I.cie();
    uint64_t GoodPC = I.CodeAddr;
    I.fde(C);
    I.CodeAddr += 0x100;
    I.FDEProgram = std::string(1, char(Op));
    I.fde(C);
    I.finish();
    auto Info = PerfJITUnwindInfo::create(I.Bytes, I.FrameAddr, 8, true);
    ASSERT_THAT_EXPECTED(Info, Succeeded());
    EXPECT_THAT_EXPECTED(Info->emitEHFrame(I.CodeAddr, I.CodeSize), Failed());
    auto Good = Info->emitEHFrame(GoodPC, I.CodeSize);
    ASSERT_THAT_EXPECTED(Good, Succeeded());
    checkOutput(*Good, I.CodeSize, {{0x80, I.CodeSize}});
  }
}

TEST(PerfJITUnwindInfoTest, TruncationsAndSingleByteMutationsRemainBounded) {
  for (StringRef Augmentation : {"zR", "zPLR"}) {
    InputCFI I;
    I.Augmentation = Augmentation.str();
    I.make();
    auto Check = [&](StringRef Bytes) {
      auto Info = PerfJITUnwindInfo::create(Bytes, I.FrameAddr, 8, true);
      if (!Info) {
        consumeError(Info.takeError());
        return;
      }
      auto Output = Info->emitEHFrame(I.CodeAddr, I.CodeSize);
      if (!Output) {
        consumeError(Output.takeError());
        return;
      }
      DWARFDebugFrame Parsed(Triple::x86_64, true, 0x80 + I.CodeSize);
      ASSERT_THAT_ERROR(Parsed.parse(DWARFDataExtractor(*Output, true, 8)),
                        Succeeded());
    };
    for (size_t Size = 0; Size != I.Bytes.size(); ++Size)
      Check(StringRef(I.Bytes).take_front(Size));
    for (size_t Offset = 0; Offset != I.Bytes.size(); ++Offset)
      for (uint8_t Byte : {0, 0x7f, 0x80, 0xff}) {
        SCOPED_TRACE(Offset);
        SCOPED_TRACE(unsigned(Byte));
        std::string Bytes = I.Bytes;
        Bytes[Offset] = char(Byte);
        Check(Bytes);
      }
  }
}

#ifdef __linux__
TEST(PerfJITLoaderTest, UnwindFailuresStillEmitDebugAndCodeLoads) {
  if (sizeof(void *) != 8 || !sys::IsLittleEndianHost)
    GTEST_SKIP() << "Unwind support currently requires little-endian ELF64";
  using namespace llvm::orc::shared;
  auto Call = [](auto Function) {
    Error Remote = Error::success();
    auto Caller = [Function](const char *Data, size_t Size) {
      return WrapperFunctionBuffer(Function(Data, Size));
    };
    Error Transport = WrapperFunction<SPSError()>::call(Caller, Remote);
    return joinErrors(std::move(Transport), std::move(Remote));
  };
  ASSERT_THAT_ERROR(Call(llvm_orc_registerJITLoaderPerfStart), Succeeded());
  std::string Path =
      "/tmp/jit-" + Twine(sys::Process::getProcessId()).str() + ".dump";
  bool Ended = false;
  auto Cleanup = scope_exit([&] {
    if (!Ended)
      consumeError(Call(llvm_orc_registerJITLoaderPerfEnd));
    EXPECT_FALSE(sys::fs::remove(Path));
  });
  char Memory[256] = {}; // Code and CFI stay within a PC-relative 32-bit range.
  const char *Code = Memory; // Recorded, but never executed.
  SmallVector<uint32_t> ExpectedKinds;
  // Valid metadata on both sides checks that an error does not poison the
  // writer. Exercise parsing, program emission, and pre-parse size failures.
  for (unsigned Variant = 0; Variant != 10; ++Variant) {
    SCOPED_TRACE(Variant);
    InputCFI I;
    I.FrameAddr = reinterpret_cast<uintptr_t>(Memory + 8);
    I.CodeAddr = reinterpret_cast<uintptr_t>(Code);
    I.CodeSize = 1;
    if (Variant == 1)
      I.Augmentation = "zRX";
    if (Variant == 3)
      I.FDEProgram = std::string(1, char(0x17));
    if (Variant == 4)
      I.FDEProgram = std::string("\x0f\x09\x03", 3) + std::string(8, '\0');
    I.make();
    if (Variant == 2)
      I.Bytes.resize(8);
    ASSERT_LE(I.Bytes.size(), sizeof(Memory) - 8);
    std::copy(I.Bytes.begin(), I.Bytes.end(), Memory + 8);
    std::string Before = I.Bytes;

    PerfJITRecordBatch Batch{};
    auto &U = Batch.UnwindingRecord;
    U.Prefix.Id = PerfJITRecordType::JIT_CODE_UNWINDING_INFO;
    U.Prefix.TotalSize = 40 + I.Bytes.size();
    U.EHFrameAddr = I.FrameAddr;
    U.UnwindDataSize = I.Bytes.size();
    if (Variant == 5) {
      U.UnwindDataSize = 1;
      U.EHFrameHdrSize = 2;
    }
    if (Variant == 6)
      U.EHFrameAddr = 0;
    if (Variant == 7)
      U.Prefix.TotalSize = 1;
    if (Variant == 8)
      U.Prefix.TotalSize = 0; // No-CFI batches must also keep debug and code.

    PerfJITCodeLoadRecord Load{};
    Load.Name = "recorded";
    Load.Prefix.Id = PerfJITRecordType::JIT_CODE_LOAD;
    Load.Prefix.TotalSize = 56 + Load.Name.size() + 1 + 1;
    Load.CodeAddr = Load.Vma = I.CodeAddr;
    Load.CodeSize = 1;
    Load.CodeIndex = Variant;
    Batch.CodeLoadRecords.push_back(Load);
    PerfJITDebugInfoRecord Debug{};
    Debug.Prefix.Id = PerfJITRecordType::JIT_CODE_DEBUG_INFO;
    Debug.Prefix.TotalSize = 32 + 16 + sizeof("test.cc");
    Debug.CodeAddr = I.CodeAddr;
    Debug.Entries.push_back({I.CodeAddr, 1, 0, "test.cc"});
    Batch.DebugInfoRecords.push_back(Debug);

    Error Remote = Error::success();
    ASSERT_THAT_ERROR(WrapperFunction<SPSError(SPSPerfJITRecordBatch)>::call(
                          [](const char *Data, size_t Size) {
                            return WrapperFunctionBuffer(
                                llvm_orc_registerJITLoaderPerfImpl(Data, Size));
                          },
                          Remote, Batch),
                      Succeeded());
    ASSERT_THAT_ERROR(std::move(Remote), Succeeded());
    EXPECT_EQ(StringRef(Memory + 8, I.Bytes.size()), Before);
    if (Variant == 0 || Variant == 9)
      ExpectedKinds.push_back(4);
    ExpectedKinds.push_back(2);
    ExpectedKinds.push_back(0);
  }
  ASSERT_THAT_ERROR(Call(llvm_orc_registerJITLoaderPerfEnd), Succeeded());
  Ended = true;
  ExpectedKinds.push_back(3);
  auto Buffer = MemoryBuffer::getFile(Path);
  ASSERT_TRUE(Buffer);
  DataExtractor Data((*Buffer)->getBuffer(), sys::IsLittleEndianHost);
  SmallVector<uint32_t> Kinds;
  uint64_t Offset = 8;
  Offset = Data.getU32(&Offset); // Header's total_size field.
  while (Offset < Data.size()) {
    uint64_t Start = Offset;
    uint32_t Kind = Data.getU32(&Offset);
    uint32_t Size = Data.getU32(&Offset);
    ASSERT_GE(Size, 16u);
    ASSERT_LE(Size, Data.size() - Start);
    Kinds.push_back(Kind);
    Offset = Start + Size;
  }
  EXPECT_EQ(Kinds, ExpectedKinds);
}
#endif

} // namespace
