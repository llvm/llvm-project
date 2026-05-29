//===- MetallibWriter.cpp - .metallib container writer ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// LLVM Module to .metallib container.
//
// metallib container layout:
//   [0:88]   MTLB header (magic + platform + filesize + 4 section descs)
//   [88:...] Section 0: entry headers (u32 count + per-entry blocks)
//   [gap]    2x ENDT (8 bytes, NOT counted in section 0 size)
//   [...]    Section 1: function list (u32(4) + ENDT)
//   [...]    Section 2: public metadata (u32(4) + ENDT)
//   [...]    Section 3: wrapped bitcode (0x0B17C0DE wrapper + raw bitcode)
//
// Entry header tags are 4-byte ASCII + 2-byte LE length + payload.
// Exception: ENDT is just 4 bytes (no length field).
//
//===----------------------------------------------------------------------===//

#include "MetallibWriter.h"
#include "BitcodeEmitter.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>

using namespace llvm;

namespace llvm {
namespace metal {

// ── Helpers ──────────────────────────────────────────────────────────────

static void writeU16(raw_ostream &OS, uint16_t V) {
  OS.write(reinterpret_cast<const char *>(&V), 2);
}
static void writeU32(raw_ostream &OS, uint32_t V) {
  OS.write(reinterpret_cast<const char *>(&V), 4);
}
static void writeU64(raw_ostream &OS, uint64_t V) {
  OS.write(reinterpret_cast<const char *>(&V), 8);
}

static void writeTag(raw_ostream &OS, const char *Name, uint16_t Len) {
  OS.write(Name, 4);
  writeU16(OS, Len);
}

static void writeENDT(raw_ostream &OS) { OS.write("ENDT", 4); }

// ── Bitcode ──────────────────────────────────────────────────────────────

// generateBitcode is now in BitcodeEmitter.cpp (emitMetalBitcode)
// It emits typed pointers using the PointeeTypeMap.

static std::vector<uint8_t> wrapBitcode(const std::vector<uint8_t> &BC) {
  std::string Buf;
  raw_string_ostream RSO(Buf);
  writeU32(RSO, 0x0B17C0DE); // wrapper magic
  writeU32(RSO, 0);          // version
  writeU32(RSO, 20);         // offset to bitcode
  writeU32(RSO, BC.size());  // bitcode size
  writeU32(RSO, 0xFFFFFFFF); // CPU type
  RSO.write(reinterpret_cast<const char *>(BC.data()), BC.size());
  RSO.flush();
  return std::vector<uint8_t>(Buf.begin(), Buf.end());
}

// ── Entry header (tags only, no size prefix, no ENDT) ────────────────────

static std::string buildEntryTags(StringRef Name, ArrayRef<uint8_t> Hash,
                                  uint64_t BitcodeSize,
                                  const MetallibOptions &Opts) {
  std::string Buf;
  raw_string_ostream OS(Buf);

  // NAME
  writeTag(OS, "NAME", Name.size() + 1);
  OS << Name;
  OS.write('\0');

  // TYPE (2 = kernel)
  writeTag(OS, "TYPE", 1);
  OS.write(char(2));

  // HASH (SHA256 of wrapped bitcode)
  writeTag(OS, "HASH", 32);
  OS.write(reinterpret_cast<const char *>(Hash.data()), 32);

  // MDSZ (u64 = bitcode section size)
  writeTag(OS, "MDSZ", 8);
  writeU64(OS, BitcodeSize);

  // OFFT (3 × u64, all zero for single-entry)
  writeTag(OS, "OFFT", 24);
  writeU64(OS, 0);
  writeU64(OS, 0);
  writeU64(OS, 0);

  // VERS (air_major=2, air_minor=8, metal_major, metal_minor)
  writeTag(OS, "VERS", 8);
  writeU16(OS, 2);
  writeU16(OS, 8);
  writeU16(OS, Opts.MetalMajor);
  writeU16(OS, Opts.MetalMinor);

  OS.flush();
  return Buf;
}

// ── Main writer ──────────────────────────────────────────────────────────

bool writeMetallib(Module &M, PointeeTypeMap &PTM, raw_ostream &OS,
                   const MetallibOptions &Opts) {
  // Emit bitcode with typed pointers (the whole point of this project)
  auto Bitcode = emitMetalBitcode(M, PTM);
  auto WrappedBC = wrapBitcode(Bitcode);
  auto Hash = SHA256::hash(ArrayRef<uint8_t>(WrappedBC));

  // Collect kernel names from !air.kernel metadata
  SmallVector<std::string, 4> KernelNames;
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    if (auto *KMD = M.getNamedMetadata("air.kernel")) {
      for (unsigned I = 0; I < KMD->getNumOperands(); I++) {
        auto *Node = KMD->getOperand(I);
        if (Node->getNumOperands() > 0) {
          if (auto *FnMD =
                  dyn_cast_if_present<ValueAsMetadata>(Node->getOperand(0))) {
            if (FnMD->getValue() == &F)
              KernelNames.push_back(F.getName().str());
          }
        }
      }
    }
  }

  if (KernelNames.empty()) {
    OS.write(reinterpret_cast<const char *>(Bitcode.data()), Bitcode.size());
    return true;
  }

  // ── Build section 0: entry headers ─────────────────────────────────────
  // Format: u32(entry_count) + per entry: u32(entry_size) + tags
  // entry_size = tags.size() + 4(ENDT) + 4(ENDT)
  // The 2 ENDTs are written AFTER section 0 (in a gap, not counted in sec0
  // size)

  std::string Sec0;
  {
    raw_string_ostream S0(Sec0);
    writeU32(S0, KernelNames.size());
    S0.flush();
  }

  // Build per-entry tag data
  SmallVector<std::string, 4> EntryTags;
  for (auto &Name : KernelNames) {
    EntryTags.push_back(buildEntryTags(Name, Hash, WrappedBC.size(), Opts));
  }

  // Append entry size + tags to Sec0
  for (auto &Tags : EntryTags) {
    std::string Entry;
    raw_string_ostream ES(Entry);
    uint32_t EntrySize = Tags.size() + 4 + 4; // tags + ENDT + ENDT
    writeU32(ES, EntrySize);
    ES << Tags;
    ES.flush();
    Sec0 += Entry;
  }

  // The 2 ENDTs per entry go in a gap after section 0
  std::string EndtGap;
  for (size_t I = 0; I < KernelNames.size(); I++) {
    EndtGap += "ENDT";
    EndtGap += "ENDT";
  }

  // Section 1 & 2: u32(4) + ENDT
  std::string Sec12;
  {
    raw_string_ostream S12(Sec12);
    writeU32(S12, 4);
    writeENDT(S12);
    S12.flush();
  }

  // ── Compute layout ─────────────────────────────────────────────────────

  uint64_t HeaderSize = 88;
  uint64_t Sec0Offset = HeaderSize;
  uint64_t Sec0Size = Sec0.size();
  uint64_t GapSize = EndtGap.size();
  uint64_t Sec1Offset = Sec0Offset + Sec0Size + GapSize;
  uint64_t Sec1Size = Sec12.size();
  uint64_t Sec2Offset = Sec1Offset + Sec1Size;
  uint64_t Sec2Size = Sec12.size();
  uint64_t Sec3Offset = Sec2Offset + Sec2Size;
  uint64_t Sec3Size = WrappedBC.size();
  uint64_t TotalSize = Sec3Offset + Sec3Size;

  // ── Write MTLB header ─────────────────────────────────────────────────

  OS.write("MTLB", 4);

  uint8_t Platform[12] = {0x01, 0x80, 0x02, 0x00, 0x09, 0x00,
                          0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  Platform[7] = static_cast<uint8_t>(Opts.Platform);
  Platform[8] = Opts.OSMajor & 0xFF;
  OS.write(reinterpret_cast<const char *>(Platform), 12);

  writeU64(OS, TotalSize);

  writeU64(OS, Sec0Offset);
  writeU64(OS, Sec0Size);
  writeU64(OS, Sec1Offset);
  writeU64(OS, Sec1Size);
  writeU64(OS, Sec2Offset);
  writeU64(OS, Sec2Size);
  writeU64(OS, Sec3Offset);
  writeU64(OS, Sec3Size);

  // ── Write body ─────────────────────────────────────────────────────────

  OS << Sec0;
  OS << EndtGap;
  OS << Sec12; // section 1
  OS << Sec12; // section 2
  OS.write(reinterpret_cast<const char *>(WrappedBC.data()), WrappedBC.size());

  return true;
}

std::vector<uint8_t> serializeMetallib(Module &M, PointeeTypeMap &PTM,
                                       const MetallibOptions &Opts) {
  std::string Buf;
  raw_string_ostream RSO(Buf);
  writeMetallib(M, PTM, RSO, Opts);
  RSO.flush();
  return std::vector<uint8_t>(Buf.begin(), Buf.end());
}

} // namespace metal
} // namespace llvm
