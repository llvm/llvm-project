//===- unittest/ProfileData/GCOVTest.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/GCOV.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#include "llvm/Support/Windows/WindowsSupport.h"
#include "llvm/Support/WindowsError.h"
#elif defined(LLVM_ON_UNIX) && !defined(__MVS__)
#include <sys/mman.h>
#endif

using namespace llvm;

namespace {

enum : uint32_t {
  GCOV_ARC_ON_TREE = 1 << 0,

  GCOV_TAG_FUNCTION = 0x01000000,
  GCOV_TAG_BLOCKS = 0x01410000,
  GCOV_TAG_ARCS = 0x01430000,
  GCOV_TAG_LINES = 0x01450000,
  GCOV_TAG_COUNTER_ARCS = 0x01a10000,
  GCOV_TAG_OBJECT_SUMMARY = 0xa1000000,
  GCOV_TAG_PROGRAM_SUMMARY = 0xa3000000,
};

StringRef versionMagicLE(GCOV::GCOVVersion Version) {
  switch (Version) {
  case GCOV::V304:
    return "*403";
  case GCOV::V407:
    return "*704";
  case GCOV::V408:
    return "*804";
  case GCOV::V800:
    return "*08A";
  case GCOV::V900:
    return "*09A";
  case GCOV::V1200:
    return " 02B";
  }
  llvm_unreachable("Unhandled GCOVVersion");
}

/// Helper to build little-endian GCOV binary payloads deterministically across
/// host architectures.
class GCOVBinaryBuilder {
public:
  explicit GCOVBinaryBuilder(GCOV::GCOVVersion Version) : Version(Version) {}

  void writeBytes(StringRef S) {
    Bytes.insert(Bytes.end(), S.bytes_begin(), S.bytes_end());
  }

  void writeU32(uint32_t V) {
    uint8_t Tmp[4];
    support::endian::write32le(Tmp, V);
    Bytes.insert(Bytes.end(), std::begin(Tmp), std::end(Tmp));
  }

  void writeU64(uint64_t V) {
    writeU32(static_cast<uint32_t>(V & 0xffffffffULL));
    writeU32(static_cast<uint32_t>(V >> 32));
  }

  void writeString(StringRef S) {
    if (S.empty()) {
      writeU32(0);
      return;
    }
    if (Version >= GCOV::V1200) {
      writeU32(static_cast<uint32_t>(S.size() + 1));
      writeBytes(S);
      Bytes.push_back(0);
    } else {
      size_t Words = (S.size() / 4) + 1;
      writeU32(static_cast<uint32_t>(Words));
      writeBytes(S);
      size_t Pad = Words * 4 - S.size();
      Bytes.insert(Bytes.end(), Pad, 0);
    }
  }

  void writeGCNOHeader(uint32_t Checksum = 0x12345678, StringRef Cwd = "/cwd",
                       uint32_t HasUnexecutedBlocks = 0) {
    writeBytes("oncg");
    writeBytes(versionMagicLE(Version));
    writeU32(Checksum);
    if (Version >= GCOV::V900)
      writeString(Cwd);
    if (Version >= GCOV::V800)
      writeU32(HasUnexecutedBlocks);
  }

  void writeGCDAHeader(uint32_t Checksum = 0x12345678) {
    writeBytes("adcg");
    writeBytes(versionMagicLE(Version));
    writeU32(Checksum);
  }

  size_t beginRecord(uint32_t Tag) {
    writeU32(Tag);
    size_t LenOffset = Bytes.size();
    writeU32(0);
    return LenOffset;
  }

  void endRecord(size_t LenOffset) {
    size_t PayloadBytes = Bytes.size() - (LenOffset + 4);
    uint32_t EncodedLen = Version >= GCOV::V1200
                              ? static_cast<uint32_t>(PayloadBytes)
                              : static_cast<uint32_t>(PayloadBytes / 4);
    support::endian::write32le(Bytes.data() + LenOffset, EncodedLen);
  }

  void writeFunctionRecord(uint32_t Ident, StringRef Name, StringRef Filename,
                           uint32_t LinenoChecksum = 0x11111111,
                           uint32_t CfgChecksum = 0x22222222,
                           uint32_t StartLine = 10) {
    size_t Off = beginRecord(GCOV_TAG_FUNCTION);
    writeU32(Ident);
    writeU32(LinenoChecksum);
    if (Version >= GCOV::V407)
      writeU32(CfgChecksum);
    writeString(Name);
    if (Version < GCOV::V800) {
      writeString(Filename);
      writeU32(StartLine);
    } else {
      writeU32(0); // artificial
      writeString(Filename);
      writeU32(StartLine);
      writeU32(1); // startColumn
      writeU32(StartLine + 5);
      if (Version >= GCOV::V900)
        writeU32(1); // endColumn
    }
    endRecord(Off);
  }

  void writeBlocksRecord(uint32_t NumBlocks) {
    size_t Off = beginRecord(GCOV_TAG_BLOCKS);
    if (Version < GCOV::V800) {
      for (uint32_t I = 0; I < NumBlocks; ++I)
        writeU32(0);
    } else {
      writeU32(NumBlocks);
    }
    endRecord(Off);
  }

  void writeArcsRecord(uint32_t SrcBlock,
                       ArrayRef<std::pair<uint32_t, uint32_t>> Arcs) {
    size_t Off = beginRecord(GCOV_TAG_ARCS);
    writeU32(SrcBlock);
    for (const auto &[Dst, Flags] : Arcs) {
      writeU32(Dst);
      writeU32(Flags);
    }
    endRecord(Off);
  }

  void writeLinesRecord(uint32_t BlockNo, StringRef Filename,
                        ArrayRef<uint32_t> Lines) {
    size_t Off = beginRecord(GCOV_TAG_LINES);
    writeU32(BlockNo);
    writeU32(0);
    writeString(Filename);
    for (uint32_t L : Lines)
      writeU32(L);
    writeU32(0);
    writeString("");
    endRecord(Off);
  }

  void writeGCDAFunctionRecord(uint32_t Ident,
                               uint32_t LinenoChecksum = 0x11111111,
                               uint32_t CfgChecksum = 0x22222222) {
    size_t Off = beginRecord(GCOV_TAG_FUNCTION);
    writeU32(Ident);
    writeU32(LinenoChecksum);
    if (Version >= GCOV::V407)
      writeU32(CfgChecksum);
    endRecord(Off);
  }

  void writeGCDACounterArcsRecord(ArrayRef<uint64_t> Counts) {
    size_t Off = beginRecord(GCOV_TAG_COUNTER_ARCS);
    for (uint64_t C : Counts)
      writeU64(C);
    endRecord(Off);
  }

  void writeGCDAObjectSummaryRecord(uint32_t Runs) {
    size_t Off = beginRecord(GCOV_TAG_OBJECT_SUMMARY);
    writeU32(Runs);
    writeU32(0);
    endRecord(Off);
  }

  ArrayRef<uint8_t> bytes() const { return Bytes; }
  std::vector<uint8_t> takeBytes() { return std::move(Bytes); }

private:
  GCOV::GCOVVersion Version;
  std::vector<uint8_t> Bytes;
};

std::unique_ptr<MemoryBuffer> createBuffer(ArrayRef<uint8_t> Bytes) {
  return MemoryBuffer::getMemBuffer(
      StringRef(reinterpret_cast<const char *>(Bytes.data()), Bytes.size()),
      "gcov-test", /*RequiresNullTerminator=*/false);
}

bool parseGCNO(ArrayRef<uint8_t> Bytes, GCOVFile &GF) {
  auto MemBuf = createBuffer(Bytes);
  GCOVBuffer GB(MemBuf.get());
  return GF.readGCNO(GB);
}

bool parseGCDA(ArrayRef<uint8_t> Bytes, GCOVFile &GF) {
  auto MemBuf = createBuffer(Bytes);
  GCOVBuffer GB(MemBuf.get());
  return GF.readGCDA(GB);
}

GCOVBinaryBuilder buildMinimalValidGCNO(GCOV::GCOVVersion Version,
                                        StringRef FuncName = "_Z3foov") {
  GCOVBinaryBuilder B(Version);
  B.writeGCNOHeader();
  B.writeFunctionRecord(/*Ident=*/1, FuncName, "foo.cpp");
  B.writeBlocksRecord(/*NumBlocks=*/3);
  // Block 0 (entry) -> Block 2 (body, measured), Block 2 -> Block 1 (exit, on
  // tree).
  B.writeArcsRecord(0, {{2, 0}});
  B.writeArcsRecord(2, {{1, GCOV_ARC_ON_TREE}});
  B.writeLinesRecord(2, "foo.cpp", {11, 12});
  return B;
}

GCOVBinaryBuilder buildMinimalValidGCDA(GCOV::GCOVVersion Version,
                                        uint64_t MeasuredCount = 7) {
  GCOVBinaryBuilder B(Version);
  B.writeGCDAHeader();
  B.writeGCDAObjectSummaryRecord(/*Runs=*/1);
  B.writeGCDAFunctionRecord(/*Ident=*/1);
  B.writeGCDACounterArcsRecord({MeasuredCount});
  return B;
}

struct GuardedMemoryBuffer {
  sys::OwningMemoryBlock Storage;
  std::unique_ptr<MemoryBuffer> Buffer;
};

constexpr bool guardPagesSupported() {
#if defined(_WIN32) || (defined(LLVM_ON_UNIX) && !defined(__MVS__))
  return true;
#else
  return false;
#endif
}

std::error_code protectGuardPage(sys::MemoryBlock GuardPage) {
#if defined(_WIN32)
  DWORD OldProtection;
  if (!VirtualProtect(GuardPage.base(), GuardPage.allocatedSize(),
                      PAGE_NOACCESS, &OldProtection))
    return mapLastWindowsError();
  return std::error_code();
#elif defined(LLVM_ON_UNIX) && !defined(__MVS__)
  if (::mprotect(GuardPage.base(), GuardPage.allocatedSize(), PROT_NONE))
    return errnoAsErrorCode();
  return std::error_code();
#else
  return std::make_error_code(std::errc::not_supported);
#endif
}

Expected<GuardedMemoryBuffer>
createGuardedMemoryBuffer(ArrayRef<uint8_t> Bytes) {
  auto PageSizeOrErr = sys::Process::getPageSize();
  if (!PageSizeOrErr)
    return PageSizeOrErr.takeError();
  const size_t PageSize = *PageSizeOrErr;
  if (Bytes.empty() || Bytes.size() > PageSize)
    return createStringError(std::errc::invalid_argument,
                             "test buffer must fit in one memory page");

  std::error_code EC;
  sys::MemoryBlock Allocation = sys::Memory::allocateMappedMemory(
      2 * PageSize, nullptr, sys::Memory::MF_READ | sys::Memory::MF_WRITE, EC);
  if (EC)
    return errorCodeToError(EC);

  sys::OwningMemoryBlock Storage(Allocation);
  auto *GuardStart = static_cast<uint8_t *>(Allocation.base()) + PageSize;
  sys::MemoryBlock GuardPage(GuardStart, PageSize);
  EC = protectGuardPage(GuardPage);
  if (EC)
    return errorCodeToError(EC);

  auto *BufferStart = GuardStart - Bytes.size();
  std::copy(Bytes.begin(), Bytes.end(), BufferStart);
  auto Buffer = createBuffer(ArrayRef<uint8_t>(BufferStart, Bytes.size()));
  return GuardedMemoryBuffer{std::move(Storage), std::move(Buffer)};
}

TEST(GCOVTest, ValidMinimalRoundTrip) {
  for (GCOV::GCOVVersion Version :
       {GCOV::V408, GCOV::V800, GCOV::V900, GCOV::V1200}) {
    // Variant 0: GCC-style implicit EOF (exact end of buffer).
    // Variant 1: Single zero tag word trailer.
    // Variant 2: Clang-style {tag = 0, length = 0} trailer.
    for (int TrailerWords : {0, 1, 2}) {
      GCOVBinaryBuilder GCNO = buildMinimalValidGCNO(Version);
      GCOVBinaryBuilder GCDA =
          buildMinimalValidGCDA(Version, /*MeasuredCount=*/9);
      for (int I = 0; I < TrailerWords; ++I) {
        GCNO.writeU32(0);
        GCDA.writeU32(0);
      }

      GCOVFile GF;
      ASSERT_TRUE(parseGCNO(GCNO.bytes(), GF));
      ASSERT_TRUE(parseGCDA(GCDA.bytes(), GF));
      EXPECT_EQ(GF.getVersion(), Version);
      EXPECT_EQ(GF.runCount, 1u);
      ASSERT_EQ(GF.functions.size(), 1u);
      const GCOVFunction &Fn = *GF.functions[0];
      EXPECT_EQ(Fn.getName(/*demangle=*/false), "_Z3foov");
      EXPECT_EQ(Fn.getName(/*demangle=*/true), "foo()");
      EXPECT_EQ(Fn.getFilename(), "foo.cpp");
      EXPECT_EQ(Fn.getEntryCount(), 9u);
    }

    // Empty function name (len == 0) in GCOV_TAG_FUNCTION is accepted.
    GCOVBinaryBuilder EmptyNameGCNO = buildMinimalValidGCNO(Version, "");
    GCOVFile EmptyNameFile;
    ASSERT_TRUE(parseGCNO(EmptyNameGCNO.bytes(), EmptyNameFile));
    ASSERT_EQ(EmptyNameFile.functions.size(), 1u);
    EXPECT_EQ(EmptyNameFile.functions[0]->getName(/*demangle=*/false), "");
  }
}

TEST(GCOVTest, TruncatedAndMalformedHeaders) {
  // Shorter than 4-byte magic, wrong magic, truncated version, and unsupported
  // version.
  for (StringRef Bad : {StringRef("onc"), StringRef("xxxx*08A12345678"),
                        StringRef("oncg*0"), StringRef("oncg*30312345678")}) {
    GCOVFile GF;
    ArrayRef<uint8_t> Bytes(reinterpret_cast<const uint8_t *>(Bad.data()),
                            Bad.size());
    EXPECT_FALSE(parseGCNO(Bytes, GF));
  }

  // Missing checksum (only 8 bytes).
  {
    GCOVFile GF;
    StringRef Trunc("oncg*804");
    EXPECT_FALSE(parseGCNO(
        ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(Trunc.data()),
                          Trunc.size()),
        GF));
  }

  // 12-byte V800 .gcno missing hasUnexecutedBlocks word (previously returned
  // true silently via unchecked getWord() + while ((tag = buf.getWord()))).
  {
    GCOVBinaryBuilder B(GCOV::V800);
    B.writeBytes("oncg");
    B.writeBytes("*08A");
    B.writeU32(0x12345678);
    GCOVFile GF;
    EXPECT_FALSE(parseGCNO(B.bytes(), GF));
    EXPECT_FALSE(GF.GCNOInitialized);
  }

  // V900 .gcno with truncated cwd string in header.
  {
    GCOVBinaryBuilder B(GCOV::V900);
    B.writeBytes("oncg");
    B.writeBytes("*09A");
    B.writeU32(0x12345678);
    B.writeU32(4); // Claims 16 bytes of cwd, but EOF follows.
    GCOVFile GF;
    EXPECT_FALSE(parseGCNO(B.bytes(), GF));
  }
}

TEST(GCOVTest, TrailingTruncatedTagAndLengthBytes) {
  for (GCOV::GCOVVersion Version : {GCOV::V800, GCOV::V1200}) {
    std::vector<uint8_t> BaseGCNO = buildMinimalValidGCNO(Version).takeBytes();
    std::vector<uint8_t> BaseGCDA = buildMinimalValidGCDA(Version).takeBytes();

    // 1, 2, or 3 trailing truncated bytes after a valid record must fail.
    for (size_t Extra = 1; Extra <= 3; ++Extra) {
      std::vector<uint8_t> BadGCNO = BaseGCNO;
      BadGCNO.insert(BadGCNO.end(), Extra, 0xaa);
      GCOVFile GF;
      EXPECT_FALSE(parseGCNO(BadGCNO, GF));

      GCOVFile ValidGF;
      ASSERT_TRUE(parseGCNO(BaseGCNO, ValidGF));
      std::vector<uint8_t> BadGCDA = BaseGCDA;
      BadGCDA.insert(BadGCDA.end(), Extra, 0xaa);
      EXPECT_FALSE(parseGCDA(BadGCDA, ValidGF));
    }

    // Non-zero tag at EOF missing its 4-byte length word must fail.
    {
      GCOVBinaryBuilder B = buildMinimalValidGCNO(Version);
      B.writeU32(GCOV_TAG_BLOCKS);
      GCOVFile GF;
      EXPECT_FALSE(parseGCNO(B.bytes(), GF));
    }
  }
}

TEST(GCOVTest, RecordLengthExceedsBufferOrOverreadsRecordEnd) {
  for (GCOV::GCOVVersion Version : {GCOV::V408, GCOV::V1200}) {
    // length = 0xffffffff (32-bit 4 * length wrap on < V1200).
    {
      GCOVBinaryBuilder B(Version);
      B.writeGCNOHeader();
      B.writeU32(GCOV_TAG_FUNCTION);
      B.writeU32(0xffffffff);
      GCOVFile GF;
      EXPECT_FALSE(parseGCNO(B.bytes(), GF));
    }

    // Record length 4 bytes larger than remaining buffer.
    {
      GCOVBinaryBuilder B(Version);
      B.writeGCNOHeader();
      B.writeU32(0x01990000); // Unknown skippable tag
      B.writeU32(Version >= GCOV::V1200 ? 8 : 2);
      B.writeU32(0); // Only 4 bytes present instead of 8.
      GCOVFile GF;
      EXPECT_FALSE(parseGCNO(B.bytes(), GF));
    }
  }

  // GCOV_TAG_FUNCTION body over-reading its declared record length into a
  // subsequent record must fail without mutating GCOVFile maps.
  {
    GCOVBinaryBuilder B(GCOV::V800);
    B.writeGCNOHeader();
    B.writeU32(GCOV_TAG_FUNCTION);
    B.writeU32(3); // Declared 12 bytes, much smaller than function payload.
    B.writeU32(1); // ident
    B.writeU32(2); // linenoChecksum
    B.writeU32(3); // cfgChecksum
    B.writeString("fn");
    B.writeU32(0); // artificial
    B.writeString("file.c");
    B.writeU32(1); // startLine
    B.writeU32(1); // startColumn
    B.writeU32(2); // endLine
    B.writeU32(0); // trailer
    B.writeU32(0);

    GCOVFile GF;
    EXPECT_FALSE(parseGCNO(B.bytes(), GF));
    EXPECT_TRUE(GF.functions.empty());
    EXPECT_TRUE(GF.filenames.empty());
    EXPECT_TRUE(GF.identToFunction.empty());
  }
}

TEST(GCOVTest, ArcsLengthUnderflowAndOddBytes) {
  // V408 length = 0 previously underflowed (length - 1) / 2 to 0x7fffffff.
  {
    GCOVBinaryBuilder B(GCOV::V408);
    B.writeGCNOHeader();
    B.writeFunctionRecord(1, "f", "f.c");
    B.writeBlocksRecord(2);
    B.writeU32(GCOV_TAG_ARCS);
    B.writeU32(0);
    GCOVFile GF;
    EXPECT_FALSE(parseGCNO(B.bytes(), GF));
  }

  // V1200 length = 0..3 previously underflowed (length / 4 - 1) / 2.
  for (uint32_t BadLen = 0; BadLen < 4; ++BadLen) {
    GCOVBinaryBuilder B(GCOV::V1200);
    B.writeGCNOHeader();
    B.writeFunctionRecord(1, "f", "f.c");
    B.writeBlocksRecord(2);
    B.writeU32(GCOV_TAG_ARCS);
    B.writeU32(BadLen);
    for (uint32_t I = 0; I < BadLen; ++I)
      B.writeBytes(StringRef("\0", 1));
    GCOVFile GF;
    EXPECT_FALSE(parseGCNO(B.bytes(), GF));
  }

  // Partial arc trailing bytes (byteLen = 8 -> srcNo + 4 leftover bytes).
  for (GCOV::GCOVVersion Version : {GCOV::V408, GCOV::V1200}) {
    GCOVBinaryBuilder B(Version);
    B.writeGCNOHeader();
    B.writeFunctionRecord(1, "f", "f.c");
    B.writeBlocksRecord(2);
    size_t Off = B.beginRecord(GCOV_TAG_ARCS);
    B.writeU32(0); // srcNo
    B.writeU32(1); // dstNo without flags word
    B.endRecord(Off);
    GCOVFile GF;
    EXPECT_FALSE(parseGCNO(B.bytes(), GF));
  }
}

TEST(GCOVTest, ArcsOutOfBoundsSrcAndDstBlock) {
  for (std::pair<uint32_t, uint32_t> BadArc :
       {std::pair{2u, 0u}, std::pair{0u, 2u}, std::pair{0u, 0xffffffffu}}) {
    GCOVBinaryBuilder B(GCOV::V800);
    B.writeGCNOHeader();
    B.writeFunctionRecord(1, "f", "f.c");
    B.writeBlocksRecord(2);
    B.writeArcsRecord(BadArc.first, {{BadArc.second, 0}});
    GCOVFile GF;
    EXPECT_FALSE(parseGCNO(B.bytes(), GF));
  }
}

TEST(GCOVTest, LinesBeforeInitialFilenameAndOutOfBoundsBlock) {
  // Out-of-bounds srcNo in GCOV_TAG_LINES.
  {
    GCOVBinaryBuilder B(GCOV::V800);
    B.writeGCNOHeader();
    B.writeFunctionRecord(1, "f", "f.c");
    B.writeBlocksRecord(2);
    B.writeLinesRecord(/*BlockNo=*/2, "f.c", {10});
    GCOVFile GF;
    EXPECT_FALSE(parseGCNO(B.bytes(), GF));
  }

  // Non-zero line number before any line == 0 filename entry (locations is
  // empty).
  {
    GCOVBinaryBuilder B(GCOV::V800);
    B.writeGCNOHeader();
    B.writeFunctionRecord(1, "f", "f.c");
    B.writeBlocksRecord(2);
    size_t Off = B.beginRecord(GCOV_TAG_LINES);
    B.writeU32(0);  // srcNo
    B.writeU32(42); // line != 0 before initial filename!
    B.writeU32(0);
    B.writeU32(0);
    B.endRecord(Off);
    GCOVFile GF;
    EXPECT_FALSE(parseGCNO(B.bytes(), GF));
  }
}

TEST(GCOVTest, LinesTruncatedAfterZeroSentinel) {
  for (GCOV::GCOVVersion Version : {GCOV::V800, GCOV::V1200}) {
    // Case 1: GCOV_TAG_LINES ending with [srcNo = 0, line = 0] at exact EOF.
    {
      GCOVBinaryBuilder B(Version);
      B.writeGCNOHeader();
      B.writeFunctionRecord(1, "f", "f.c");
      B.writeBlocksRecord(2);
      size_t Off = B.beginRecord(GCOV_TAG_LINES);
      B.writeU32(0); // srcNo
      B.writeU32(0); // line == 0 sentinel with missing filename length!
      B.endRecord(Off);
      GCOVFile GF;
      EXPECT_FALSE(parseGCNO(B.bytes(), GF));
    }

    // Case 2: Same truncated GCOV_TAG_LINES followed by {0, 0} trailer so the
    // underlying buffer is not at EOF, testing endPos enforcement before
    // readString.
    {
      GCOVBinaryBuilder B(Version);
      B.writeGCNOHeader();
      B.writeFunctionRecord(1, "f", "f.c");
      B.writeBlocksRecord(2);
      size_t Off = B.beginRecord(GCOV_TAG_LINES);
      B.writeU32(0); // srcNo
      B.writeU32(0); // line == 0 sentinel at endPos
      B.endRecord(Off);
      B.writeU32(0);
      B.writeU32(0);
      GCOVFile GF;
      EXPECT_FALSE(parseGCNO(B.bytes(), GF));
    }
  }
}

TEST(GCOVTest, BlocksDuplicateTagAndHugeCountOOM) {
  // Duplicate GCOV_TAG_BLOCKS for the same function must fail.
  {
    GCOVBinaryBuilder B(GCOV::V800);
    B.writeGCNOHeader();
    B.writeFunctionRecord(1, "f", "f.c");
    B.writeBlocksRecord(2);
    B.writeBlocksRecord(2);
    GCOVFile GF;
    EXPECT_FALSE(parseGCNO(B.bytes(), GF));
  }

  // V800 GCOV_TAG_BLOCKS with 0-byte payload (missing block count word).
  {
    GCOVBinaryBuilder B(GCOV::V800);
    B.writeGCNOHeader();
    B.writeFunctionRecord(1, "f", "f.c");
    size_t Off = B.beginRecord(GCOV_TAG_BLOCKS);
    B.endRecord(Off);
    GCOVFile GF;
    EXPECT_FALSE(parseGCNO(B.bytes(), GF));
  }

  // Single-function V800 num = 0xffffffff OOM amplification must fail
  // immediately without allocating gigabytes.
  {
    GCOVBinaryBuilder B(GCOV::V800);
    B.writeGCNOHeader();
    B.writeFunctionRecord(1, "f", "f.c");
    B.writeBlocksRecord(0xffffffffu);
    GCOVFile GF;
    EXPECT_FALSE(parseGCNO(B.bytes(), GF));
  }

  // Multi-function cumulative cap: each function requests 800 blocks
  // (individually below de.size() + 1024), but 4 functions (3200 blocks) exceed
  // maxBlocks.
  {
    GCOVBinaryBuilder B(GCOV::V800);
    B.writeGCNOHeader();
    for (uint32_t Id = 1; Id <= 4; ++Id) {
      B.writeFunctionRecord(Id, "f", "f.c");
      B.writeBlocksRecord(800);
    }
    GCOVFile GF;
    EXPECT_FALSE(parseGCNO(B.bytes(), GF));
  }
}

TEST(GCOVTest, ReadStringMalformedLengthsAndMissingNul) {
  // V1200 len exceeding remaining buffer (previously UB drop_back() on empty
  // StringRef) and sticky cursor state on subsequent readInt.
  {
    GCOVBinaryBuilder B(GCOV::V1200);
    B.writeBytes("oncg");
    B.writeBytes(" 02B");
    B.writeU32(100); // String length > remaining bytes
    B.writeU32(42);  // Trailing word that must not be read after cursor fails
    auto MemBuf = createBuffer(B.bytes());
    GCOVBuffer GB(MemBuf.get());
    ASSERT_TRUE(GB.readGCNOFormat());
    GCOV::GCOVVersion Ver;
    ASSERT_TRUE(GB.readGCOVVersion(Ver));
    StringRef S;
    EXPECT_FALSE(GB.readString(S));
    EXPECT_FALSE(bool(GB.cursor));
    uint32_t Next = 123;
    EXPECT_FALSE(GB.readInt(Next));
    EXPECT_EQ(Next, 0u);
  }

  // V1200 string missing trailing NUL byte in GCOVBuffer::readString and inside
  // GCOV_TAG_LINES (must fail rather than masquerading as empty terminator).
  {
    GCOVBinaryBuilder B(GCOV::V1200);
    B.writeGCNOHeader();
    B.writeFunctionRecord(1, "f", "f.c");
    B.writeBlocksRecord(2);
    size_t Off = B.beginRecord(GCOV_TAG_LINES);
    B.writeU32(0);       // srcNo
    B.writeU32(0);       // line == 0
    B.writeU32(3);       // len = 3 bytes
    B.writeBytes("abc"); // Missing trailing '\0'!
    B.endRecord(Off);
    GCOVFile GF;
    EXPECT_FALSE(parseGCNO(B.bytes(), GF));
  }

  // V408 len = 0x40000000 (uint32_t len * 4 wraps to 0) in both readString()
  // and getString().
  for (bool UseGetString : {false, true}) {
    GCOVBinaryBuilder B(GCOV::V408);
    B.writeBytes("oncg");
    B.writeBytes("*804");
    B.writeU32(0x40000000u);
    auto MemBuf = createBuffer(B.bytes());
    GCOVBuffer GB(MemBuf.get());
    ASSERT_TRUE(GB.readGCNOFormat());
    GCOV::GCOVVersion Ver;
    ASSERT_TRUE(GB.readGCOVVersion(Ver));
    if (UseGetString) {
      EXPECT_TRUE(GB.getString().empty());
    } else {
      StringRef S;
      EXPECT_FALSE(GB.readString(S));
    }
    EXPECT_FALSE(bool(GB.cursor));
  }
}

TEST(GCOVTest, DemangleNonNullTerminatedBuffer) {
  // 1. Guard-page test: "_Z3fooii" ("foo(int, int)", 8 bytes) placed at the
  // very end of a readable page immediately before a PROT_NONE page.
  if (guardPagesSupported()) {
    StringRef Mangled = "_Z3fooii";
    ArrayRef<uint8_t> Raw(reinterpret_cast<const uint8_t *>(Mangled.data()),
                          Mangled.size());
    Expected<GuardedMemoryBuffer> Guarded = createGuardedMemoryBuffer(Raw);
    ASSERT_THAT_ERROR(Guarded.takeError(), Succeeded());

    GCOVFile GF;
    GCOVFunction Fn(GF);
    Fn.Name = Guarded->Buffer->getBuffer();
    EXPECT_EQ(Fn.getName(/*demangle=*/true), "foo(int, int)");
  }

  // 2. End-to-end < V1200 readGCNO test: 8-byte mangled name "_Z3fooii" stored
  // as 2 words with zero NUL padding followed immediately by valid Itanium type
  // characters ('i') in the artificial field.
  {
    GCOVBinaryBuilder B(GCOV::V800);
    B.writeGCNOHeader();
    size_t Off = B.beginRecord(GCOV_TAG_FUNCTION);
    B.writeU32(1);          // ident
    B.writeU32(0x11111111); // linenoChecksum
    B.writeU32(0x22222222); // cfgChecksum
    B.writeU32(2);          // 2 words = 8 bytes, no NUL padding!
    B.writeBytes("_Z3fooii");
    B.writeU32(0x69696969); // artificial = ASCII "iiii"
    B.writeString("foo.cpp");
    B.writeU32(1); // startLine
    B.writeU32(1); // startColumn
    B.writeU32(2); // endLine
    B.endRecord(Off);

    GCOVFile GF;
    ASSERT_TRUE(parseGCNO(B.bytes(), GF));
    ASSERT_EQ(GF.functions.size(), 1u);
    EXPECT_EQ(GF.functions[0]->getName(/*demangle=*/true), "foo(int, int)");
  }
}

TEST(GCOVTest, ReadGcdaMalformedAndStaleFunctionState) {
  GCOVBinaryBuilder ValidGCNO = buildMinimalValidGCNO(GCOV::V1200);

  // Version mismatch and checksum mismatch.
  {
    GCOVFile GF;
    ASSERT_TRUE(parseGCNO(ValidGCNO.bytes(), GF));
    GCOVBinaryBuilder WrongVerGCDA = buildMinimalValidGCDA(GCOV::V800);
    EXPECT_FALSE(parseGCDA(WrongVerGCDA.bytes(), GF));

    GCOVBinaryBuilder WrongSumGCDA(GCOV::V1200);
    WrongSumGCDA.writeGCDAHeader(/*Checksum=*/0xdeadbeef);
    EXPECT_FALSE(parseGCDA(WrongSumGCDA.bytes(), GF));
  }

  // V1200 GCOV_TAG_FUNCTION with length = 4 bytes (< 12 required bytes).
  {
    GCOVFile GF;
    ASSERT_TRUE(parseGCNO(ValidGCNO.bytes(), GF));
    GCOVBinaryBuilder B(GCOV::V1200);
    B.writeGCDAHeader();
    size_t Off = B.beginRecord(GCOV_TAG_FUNCTION);
    B.writeU32(1); // Only 4 bytes instead of 12
    B.endRecord(Off);
    EXPECT_FALSE(parseGCDA(B.bytes(), GF));
  }

  // Truncated summary records (OBJECT_SUMMARY < 8 bytes, PROGRAM_SUMMARY < 12).
  for (uint32_t Tag : {GCOV_TAG_OBJECT_SUMMARY, GCOV_TAG_PROGRAM_SUMMARY}) {
    GCOVFile GF;
    ASSERT_TRUE(parseGCNO(ValidGCNO.bytes(), GF));
    GCOVBinaryBuilder B(GCOV::V1200);
    B.writeGCDAHeader();
    size_t Off = B.beginRecord(Tag);
    B.writeU32(1); // Only 4 bytes
    B.endRecord(Off);
    EXPECT_FALSE(parseGCDA(B.bytes(), GF));
  }

  // Stale fn reset: valid function 1 followed by:
  //   (a) placeholder GCOV_TAG_FUNCTION (length = 0) + GCOV_TAG_COUNTER_ARCS,
  //   (b) unknown-ident GCOV_TAG_FUNCTION (ident = 999) +
  //   GCOV_TAG_COUNTER_ARCS, (c) duplicate GCOV_TAG_COUNTER_ARCS directly after
  //   function 1's counters.
  // None of these may overwrite or double-count function 1's entry count.
  {
    GCOVFile GF;
    ASSERT_TRUE(parseGCNO(ValidGCNO.bytes(), GF));
    GCOVBinaryBuilder B(GCOV::V1200);
    B.writeGCDAHeader();
    B.writeGCDAObjectSummaryRecord(1);
    B.writeGCDAFunctionRecord(/*Ident=*/1);
    B.writeGCDACounterArcsRecord({11});
    // (c) Stray duplicate counter record right after valid counter record:
    B.writeGCDACounterArcsRecord({99});
    // (a) Placeholder length == 0 function record + counter record:
    B.writeU32(GCOV_TAG_FUNCTION);
    B.writeU32(0);
    B.writeGCDACounterArcsRecord({77});
    // (b) Unknown ident == 999 function record + counter record:
    B.writeGCDAFunctionRecord(/*Ident=*/999);
    B.writeGCDACounterArcsRecord({55});

    ASSERT_TRUE(parseGCDA(B.bytes(), GF));
    ASSERT_EQ(GF.functions.size(), 1u);
    EXPECT_EQ(GF.functions[0]->getEntryCount(), 11u);
  }
}

} // namespace
