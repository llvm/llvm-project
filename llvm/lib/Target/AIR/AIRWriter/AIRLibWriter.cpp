//===- AIRLibWriter.cpp - .metallib container writer ----------*- C++ -*-===//
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

#include "AIRLibWriter.h"
#include "BitcodeEmitter.h"
#include "ConstantExprLower.h"
#include "PointeeTypeMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
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

// generateBitcode is now in BitcodeEmitter.cpp (emitAIRBitcode)
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
  size_t SectionSoFar = 20u + BC.size();
  size_t Pad = (16u - (SectionSoFar % 16u)) % 16u;
  if (Pad == 0)
    Pad = 16;
  for (size_t I = 0; I < Pad; ++I)
    RSO.write('\0');
  RSO.flush();
  return std::vector<uint8_t>(Buf.begin(), Buf.end());
}

// ── Entry header (tags only, no size prefix, no ENDT) ────────────────────

static std::string buildEntryTags(StringRef Name, ArrayRef<uint8_t> Hash,
                                  uint64_t BitcodeSize, uint64_t Offt0,
                                  uint64_t Offt1, uint64_t Offt2,
                                  const AIRLibOptions &Opts) {
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

  // OFFT (3 × u64); OFFT[2] = cumulative blob offset within section 3
  writeTag(OS, "OFFT", 24);
  writeU64(OS, Offt0);
  writeU64(OS, Offt1);
  writeU64(OS, Offt2);

  // VERS (air_major=2, air_minor=8, metal_major, metal_minor)
  writeTag(OS, "VERS", 8);
  writeU16(OS, 2);
  writeU16(OS, 8);
  writeU16(OS, Opts.AIRMajor);
  writeU16(OS, Opts.AIRMinor);

  OS.flush();
  return Buf;
}

// ── Kernel collection ────────────────────────────────────────────────────

// Collect kernel names in !air.kernel metadata order.
static SmallVector<std::string, 4> collectKernelNames(Module &M) {
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
  return KernelNames;
}

// Clone M into a standalone module containing only kernel Keep (plus its
// dependencies), with every other kernel dropped from !air.kernel so the
// downstream emitter sees a single-kernel module.
static std::unique_ptr<Module> cloneSingleKernel(const Module &M,
                                                 StringRef Keep,
                                                 ArrayRef<std::string> AllKernels) {
  StringSet<> DropNames;
  for (auto &N : AllKernels)
    if (N != Keep)
      DropNames.insert(N);

  ValueToValueMapTy VMap;
  auto Clone = CloneModule(M, VMap, [&](const GlobalValue *GV) {
    if (auto *F = dyn_cast<Function>(GV))
      if (F->hasName() && DropNames.count(F->getName()))
        return false;
    return true;
  });

  // Strip the dropped kernels from !air.kernel before erasing their functions
  // (the nodes reference those functions via ValueAsMetadata).
  if (auto *KMD = Clone->getNamedMetadata("air.kernel")) {
    SmallVector<MDNode *, 4> Keepers;
    for (unsigned I = 0; I < KMD->getNumOperands(); I++) {
      auto *Node = KMD->getOperand(I);
      bool IsKeep = false;
      if (Node->getNumOperands() > 0)
        if (auto *FnMD =
                dyn_cast_if_present<ValueAsMetadata>(Node->getOperand(0)))
          if (auto *F = dyn_cast<Function>(FnMD->getValue()))
            IsKeep = F->getName() == Keep;
      if (IsKeep)
        Keepers.push_back(Node);
    }
    KMD->clearOperands();
    for (auto *Node : Keepers)
      KMD->addOperand(Node);
  }

  for (auto &Entry : DropNames)
    if (auto *F = Clone->getFunction(Entry.getKey())) {
      F->replaceAllUsesWith(UndefValue::get(F->getType()));
      F->eraseFromParent();
    }

  return Clone;
}

// ── Main writer ──────────────────────────────────────────────────────────

// One standalone wrapped-bitcode blob per kernel, concatenated in section 3.
// Each entry's MDSZ is its own blob span and OFFT[2] its cumulative offset
// within section 3; entries are packed with a single ENDT separator (omitted
// after the last) and the entry list is terminated by two trailing ENDTs in a
// gap not counted in Sec0Size. This layout is what Apple emits for any kernel
// count, so the single-kernel case flows through it unchanged.
bool writeAIRLib(Module &M, PointeeTypeMap &PTM, raw_ostream &OS,
                   const AIRLibOptions &Opts) {
  SmallVector<std::string, 4> KernelNames = collectKernelNames(M);

  if (KernelNames.empty()) {
    auto Bitcode = emitAIRBitcode(M, PTM);
    OS.write(reinterpret_cast<const char *>(Bitcode.data()), Bitcode.size());
    return true;
  }

  // ── Section 3 + per-entry tags ─────────────────────────────────────────
  // For a single kernel, emit from the module directly with the caller's PTM;
  // for several, clone each kernel into its own module so the emitter sees a
  // single-kernel module per blob.
  bool Single = KernelNames.size() == 1;
  std::string Sec3;
  SmallVector<std::string, 4> EntryTags;
  uint64_t CumOffset = 0;
  for (size_t I = 0; I < KernelNames.size(); I++) {
    std::vector<uint8_t> Bitcode;
    if (Single) {
      Bitcode = emitAIRBitcode(M, PTM);
    } else {
      auto Clone = cloneSingleKernel(M, KernelNames[I], KernelNames);
      lowerConstantExprs(*Clone);
      PointeeTypeMap ClonePTM = buildPointeeTypeMap(*Clone);
      Bitcode = emitAIRBitcode(*Clone, ClonePTM);
    }
    auto WrappedBC = wrapBitcode(Bitcode);
    auto Hash = SHA256::hash(ArrayRef<uint8_t>(WrappedBC));

    uint64_t Intra = I == 0 ? 0 : 8;
    EntryTags.push_back(buildEntryTags(KernelNames[I], Hash, WrappedBC.size(),
                                       Intra, Intra, CumOffset, Opts));
    Sec3.append(reinterpret_cast<const char *>(WrappedBC.data()),
                WrappedBC.size());
    CumOffset += WrappedBC.size();
  }

  // ── Build section 0 ────────────────────────────────────────────────────
  // u32(count) then per entry: u32(blockSize) + tags + (ENDT separator, all
  // but the last entry). blockSize counts the 4-byte size field + tags + the
  // 4-byte ENDT.
  std::string Sec0;
  {
    raw_string_ostream S0(Sec0);
    writeU32(S0, KernelNames.size());
    S0.flush();
  }
  for (size_t I = 0; I < EntryTags.size(); I++) {
    std::string Entry;
    raw_string_ostream ES(Entry);
    uint32_t BlockSize = 4 + EntryTags[I].size() + 4; // size field + tags + ENDT
    writeU32(ES, BlockSize);
    ES << EntryTags[I];
    if (I + 1 < EntryTags.size())
      writeENDT(ES);
    ES.flush();
    Sec0 += Entry;
  }

  // Two ENDTs terminate section 0; they sit in a gap after the section and are
  // NOT counted in Sec0Size (Apple writes exactly two, regardless of N).
  std::string EndtGap;
  {
    raw_string_ostream EG(EndtGap);
    writeENDT(EG);
    writeENDT(EG);
    EG.flush();
  }

  // Section 1 & 2: one (u32(4) + ENDT) pair per kernel.
  std::string Sec12;
  {
    raw_string_ostream S12(Sec12);
    for (size_t I = 0; I < KernelNames.size(); I++) {
      writeU32(S12, 4);
      writeENDT(S12);
    }
    S12.flush();
  }

  // ── Compute layout ─────────────────────────────────────────────────────
  uint64_t HeaderSize = 88;
  uint64_t Sec0Offset = HeaderSize;
  uint64_t Sec0Size = Sec0.size();
  uint64_t Sec1Offset = Sec0Offset + Sec0Size + EndtGap.size();
  uint64_t Sec1Size = Sec12.size();
  uint64_t Sec2Offset = Sec1Offset + Sec1Size;
  uint64_t Sec2Size = Sec12.size();
  uint64_t Sec3Offset = Sec2Offset + Sec2Size;
  uint64_t Sec3Size = Sec3.size();
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
  OS.write(Sec3.data(), Sec3.size());

  return true;
}

std::vector<uint8_t> serializeAIRLib(Module &M, PointeeTypeMap &PTM,
                                       const AIRLibOptions &Opts) {
  std::string Buf;
  raw_string_ostream RSO(Buf);
  writeAIRLib(M, PTM, RSO, Opts);
  RSO.flush();
  return std::vector<uint8_t>(Buf.begin(), Buf.end());
}

} // namespace metal
} // namespace llvm
