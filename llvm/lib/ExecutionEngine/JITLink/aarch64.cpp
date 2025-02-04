//===---- aarch64.cpp - Generic JITLink aarch64 edge kinds, utilities -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic utilities for graphs representing aarch64 objects.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/aarch64.h"

#include "llvm/Support/BinaryStreamWriter.h"

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {
namespace aarch64 {

const char NullPointerContent[8] = {0x00, 0x00, 0x00, 0x00,
                                    0x00, 0x00, 0x00, 0x00};

const char PointerJumpStubContent[12] = {
    0x10, 0x00, 0x00, (char)0x90u, // ADRP x16, <imm>@page21
    0x10, 0x02, 0x40, (char)0xf9u, // LDR x16, [x16, <imm>@pageoff12]
    0x00, 0x02, 0x1f, (char)0xd6u  // BR  x16
};

const char ReentryTrampolineContent[8] = {
    (char)0xfd, 0x7b, (char)0xbf, (char)0xa9, // STP x30, [sp, #-8]
    0x00,       0x00, 0x00,       (char)0x94  // BL
};

const char *getEdgeKindName(Edge::Kind R) {
  switch (R) {
  case Pointer64:
    return "Pointer64";
  case Pointer64Authenticated:
    return "Pointer64Authenticated";
  case Pointer32:
    return "Pointer32";
  case Delta64:
    return "Delta64";
  case Delta32:
    return "Delta32";
  case NegDelta64:
    return "NegDelta64";
  case NegDelta32:
    return "NegDelta32";
  case Branch26PCRel:
    return "Branch26PCRel";
  case MoveWide16:
    return "MoveWide16";
  case LDRLiteral19:
    return "LDRLiteral19";
  case TestAndBranch14PCRel:
    return "TestAndBranch14PCRel";
  case CondBranch19PCRel:
    return "CondBranch19PCRel";
  case ADRLiteral21:
    return "ADRLiteral21";
  case Page21:
    return "Page21";
  case PageOffset12:
    return "PageOffset12";
  case GotPageOffset15:
    return "GotPageOffset15";
  case RequestGOTAndTransformToPage21:
    return "RequestGOTAndTransformToPage21";
  case RequestGOTAndTransformToPageOffset12:
    return "RequestGOTAndTransformToPageOffset12";
  case RequestGOTAndTransformToPageOffset15:
    return "RequestGOTAndTransformToPageOffset15";
  case RequestGOTAndTransformToDelta32:
    return "RequestGOTAndTransformToDelta32";
  case RequestTLVPAndTransformToPage21:
    return "RequestTLVPAndTransformToPage21";
  case RequestTLVPAndTransformToPageOffset12:
    return "RequestTLVPAndTransformToPageOffset12";
  case RequestTLSDescEntryAndTransformToPage21:
    return "RequestTLSDescEntryAndTransformToPage21";
  case RequestTLSDescEntryAndTransformToPageOffset12:
    return "RequestTLSDescEntryAndTransformToPageOffset12";
  default:
    return getGenericEdgeKindName(static_cast<Edge::Kind>(R));
  }
}

// Write a 64-bit GPR -> GPR move.
template <typename AppendFtor>
static Error writeMovRegRegSeq(AppendFtor &Append, uint64_t DstReg,
                               uint64_t SrcReg) {
  assert(DstReg < 32 && "Dst reg out of range");
  assert(SrcReg < 32 && "Src reg out of range");

  if (DstReg == SrcReg)
    return Error::success();

  constexpr uint32_t MOVGPR64Template = 0xaa0003e0;
  constexpr uint32_t DstRegIndex = 0;
  constexpr uint32_t SrcRegIndex = 16;
  uint32_t Instr = MOVGPR64Template;
  Instr |= DstReg << DstRegIndex;
  Instr |= SrcReg << SrcRegIndex;
  return Append(Instr);
}

// Generate a sequence of imm writes to assign the given value.
template <typename AppendFtor>
static Error writeMovRegImm64Seq(AppendFtor &Append, uint64_t Reg,
                                 uint64_t Imm) {
  assert(Reg < 32 && "Invalid register number");

  constexpr uint32_t MovRegImm64Template = 0xd2800000;
  constexpr unsigned PreserveBitIndex = 29;
  constexpr unsigned ShiftBitsIndex = 21;
  constexpr unsigned ImmBitsIndex = 5;

  bool PreserveRegValue = false;
  for (unsigned I = 0; I != 4; ++I) {
    uint32_t ImmBits = Imm & 0xffff;
    Imm >>= 16;

    // Skip any all-zero immediates after the first one.
    if (PreserveRegValue && !ImmBits)
      continue;

    uint32_t Instr = MovRegImm64Template;
    Instr |= PreserveRegValue << PreserveBitIndex;
    Instr |= (I << ShiftBitsIndex);
    Instr |= ImmBits << ImmBitsIndex;
    Instr |= Reg;
    if (auto Err = Append(Instr))
      return Err;
    PreserveRegValue = true;
  }

  return Error::success();
}

template <typename AppendFtor>
static Error
writePACSignSeq(AppendFtor &Append, unsigned DstReg, orc::ExecutorAddr RawAddr,
                unsigned RawAddrReg, unsigned DiscriminatorReg, unsigned Key,
                uint64_t EncodedDiscriminator, bool AddressDiversify) {
  assert(DstReg < 32 && "DstReg out of range");
  assert(RawAddrReg < 32 && "AddrReg out of range");
  assert(DiscriminatorReg < 32 && "DiscriminatorReg out of range");
  assert(EncodedDiscriminator < 0x10000 && "EncodedDiscriminator out of range");

  if (AddressDiversify) {
    // Move the address into the discriminator register.
    if (auto Err = writeMovRegRegSeq(Append, DiscriminatorReg, RawAddrReg))
      return Err;
    // Blend encoded discriminator if there is one.
    if (EncodedDiscriminator) {
      constexpr uint32_t MOVKTemplate = 0xf2e00000;
      constexpr unsigned ImmIndex = 5;
      uint32_t BlendInstr = MOVKTemplate;
      BlendInstr |= EncodedDiscriminator << ImmIndex;
      BlendInstr |= DiscriminatorReg;
      if (auto Err = Append(BlendInstr))
        return Err;
    }
  } else if (EncodedDiscriminator) {
    // Move the encoded discriminator into the discriminator register.
    if (auto Err =
            writeMovRegImm64Seq(Append, DiscriminatorReg, EncodedDiscriminator))
      return Err;
  } else
    DiscriminatorReg = 31; // WZR

  constexpr uint32_t PACTemplate = 0xdac10000;
  constexpr unsigned ZBitIndex = 13;
  constexpr unsigned KeyIndex = 10;
  constexpr unsigned DiscriminatorRegIndex = 5;

  uint32_t Instr = PACTemplate;
  Instr |= (DiscriminatorReg == 31) << ZBitIndex;
  Instr |= Key << KeyIndex;
  Instr |= DiscriminatorReg << DiscriminatorRegIndex;
  Instr |= DstReg;

  return Append(Instr);
}

template <typename AppendFtor>
static Error writeStoreRegSeq(AppendFtor &Append, unsigned DstLocReg,
                              unsigned SrcReg) {
  assert(DstLocReg < 32 && "DstLocReg out of range");
  assert(SrcReg < 32 && "SrcReg out of range");

  constexpr uint32_t STRTemplate = 0xf9000000;
  constexpr unsigned DstLocRegIndex = 5;
  constexpr unsigned SrcRegIndex = 0;

  uint32_t Instr = STRTemplate;
  Instr |= DstLocReg << DstLocRegIndex;
  Instr |= SrcReg << SrcRegIndex;

  return Append(Instr);
}

void GOTTableManager::registerExistingEntries() {
  for (auto *EntrySym : GOTSection->symbols()) {
    assert(EntrySym->getBlock().edges_size() == 1 &&
           "GOT block edge count != 1");
    registerPreExistingEntry(EntrySym->getBlock().edges().begin()->getTarget(),
                             *EntrySym);
  }
}

void PLTTableManager::registerExistingEntries() {
  for (auto *EntrySym : StubsSection->symbols()) {
    assert(EntrySym->getBlock().edges_size() == 2 &&
           "PLT block edge count != 2");
    auto &GOTSym = EntrySym->getBlock().edges().begin()->getTarget();
    assert(GOTSym.getBlock().edges_size() == 1 && "GOT block edge count != 1");
    registerPreExistingEntry(GOTSym.getBlock().edges().begin()->getTarget(),
                             *EntrySym);
  }
}

const char *getPointerSigningFunctionSectionName() { return "$__ptrauth_sign"; }

/// Creates a pointer signing function section, block, and symbol to reserve
/// space for a signing function for this LinkGraph. Clients should insert this
/// pass in the post-prune phase, and add the paired
/// lowerPointer64AuthEdgesToSigningFunction pass to the pre-fixup phase.
Error createEmptyPointerSigningFunction(LinkGraph &G) {
  LLVM_DEBUG({
    dbgs() << "Creating empty pointer signing function for " << G.getName()
           << "\n";
  });

  // FIXME: We could put a tighter bound on this if we inspected the ptrauth
  // info encoded in the addend -- the only actually unknown quantity is the
  // fixup location, and we can probably put constraints even on that.
  size_t NumPtrAuthFixupLocations = 0;
  for (auto *B : G.blocks())
    for (auto &E : B->edges())
      NumPtrAuthFixupLocations +=
          E.getKind() == aarch64::Pointer64Authenticated;

  constexpr size_t MaxPtrSignSeqLength =
      4 + // To materialize the value to sign.
      4 + // To materialize the fixup location.
      3 + // To copy, blend discriminator, and sign
      1;  // To store the result.

  // The maximum number of signing instructions required is the maximum per
  // location, times the number of locations, plus three instructions to
  // materialize the return value and return.
  size_t NumSigningInstrs = NumPtrAuthFixupLocations * MaxPtrSignSeqLength + 3;

  // Create signing function section.
  auto &SigningSection =
      G.createSection(getPointerSigningFunctionSectionName(),
                      orc::MemProt::Read | orc::MemProt::Exec);
  SigningSection.setMemLifetime(orc::MemLifetime::Finalize);

  size_t SigningFunctionSize = NumSigningInstrs * 4;
  auto &SigningFunctionBlock = G.createMutableContentBlock(
      SigningSection, G.allocateBuffer(SigningFunctionSize),
      orc::ExecutorAddr(), 4, 0);
  G.addAnonymousSymbol(SigningFunctionBlock, 0, SigningFunctionBlock.getSize(),
                       true, true);

  LLVM_DEBUG({
    dbgs() << "  " << NumPtrAuthFixupLocations << " location(s) to sign, up to "
           << NumSigningInstrs << " instructions required ("
           << formatv("{0:x}", SigningFunctionBlock.getSize()) << " bytes)\n";
  });

  return Error::success();
}

/// Given a LinkGraph containing Pointer64Auth edges, transform those edges to
/// Pointer64 and add code to sign the pointers in the executor.
///
/// This function will add a $__ptrauth_sign section with finalization-lifetime
/// containing an anonymous function that will sign all pointers in the graph.
/// An allocation action will be added to run this function during finalization.
Error lowerPointer64AuthEdgesToSigningFunction(LinkGraph &G) {
  LLVM_DEBUG({
    dbgs() << "Writing pointer signing function for " << G.getName() << "\n";
  });

  constexpr unsigned Reg1 = 8;  // Holds pointer value to sign.
  constexpr unsigned Reg2 = 9;  // Holds fixup address.
  constexpr unsigned Reg3 = 10; // Temporary for discriminator value if needed.

  // Find the signing function.
  auto *SigningSection =
      G.findSectionByName(getPointerSigningFunctionSectionName());
  assert(SigningSection && "Siging section missing");
  assert(SigningSection->blocks_size() == 1 &&
         "Unexpected number of blocks in signing section");
  assert(SigningSection->symbols_size() == 1 &&
         "Unexpected number of symbols in signing section");

  auto &SigningFunctionSym = **SigningSection->symbols().begin();
  auto &SigningFunctionBlock = SigningFunctionSym.getBlock();
  auto SigningFunctionBuf = SigningFunctionBlock.getAlreadyMutableContent();

  // Write the instructions to the block content.
  BinaryStreamWriter InstrWriter(
      {reinterpret_cast<uint8_t *>(SigningFunctionBuf.data()),
       SigningFunctionBuf.size()},
      G.getEndianness());

  auto AppendInstr = [&](uint32_t Instr) {
    return InstrWriter.writeInteger(Instr);
  };

  for (auto *B : G.blocks()) {
    for (auto &E : B->edges()) {
      // We're only concerned with Pointer64Authenticated edges here.
      if (E.getKind() != aarch64::Pointer64Authenticated)
        continue;

      uint64_t EncodedInfo = E.getAddend();
      int32_t RealAddend = (uint32_t)(EncodedInfo & 0xffffffff);
      uint32_t InitialDiscriminator = (EncodedInfo >> 32) & 0xffff;
      bool AddressDiversify = (EncodedInfo >> 48) & 0x1;
      uint32_t Key = (EncodedInfo >> 49) & 0x3;
      uint32_t HighBits = EncodedInfo >> 51;
      auto ValueToSign = E.getTarget().getAddress() + RealAddend;

      if (HighBits != 0x1000)
        return make_error<JITLinkError>(
            "Pointer64Auth edge at " +
            formatv("{0:x}", B->getFixupAddress(E).getValue()) +
            " has invalid encoded addend  " + formatv("{0:x}", EncodedInfo));

      LLVM_DEBUG({
        const char *const KeyNames[] = {"IA", "IB", "DA", "DB"};
        dbgs() << "  " << B->getFixupAddress(E) << " <- " << ValueToSign
               << " : key = " << KeyNames[Key] << ", discriminator = "
               << formatv("{0:x4}", InitialDiscriminator)
               << ", address diversified = "
               << (AddressDiversify ? "yes" : "no") << "\n";
      });

      // Materialize pointer value.
      cantFail(writeMovRegImm64Seq(AppendInstr, Reg1, ValueToSign.getValue()));

      // Materialize fixup pointer.
      cantFail(writeMovRegImm64Seq(AppendInstr, Reg2,
                                   B->getFixupAddress(E).getValue()));

      // Write signing instruction(s).
      cantFail(writePACSignSeq(AppendInstr, Reg1, ValueToSign, Reg2, Reg3, Key,
                               InitialDiscriminator, AddressDiversify));

      // Store signed pointer.
      cantFail(writeStoreRegSeq(AppendInstr, Reg2, Reg1));

      // Replace edge with a keep-alive to preserve dependence info.
      E.setKind(Edge::KeepAlive);
    }
  }

  // Write epilogue. x0 = 0, x1 = 1 is an SPS serialized Error::success value.
  constexpr uint32_t RETInstr = 0xd65f03c0;
  cantFail(writeMovRegImm64Seq(AppendInstr, 0, 0)); // mov x0, #0
  cantFail(writeMovRegImm64Seq(AppendInstr, 1, 1)); // mov x1, #1
  cantFail(AppendInstr(RETInstr));                  // ret

  // Add an allocation action to call the signing function.
  using namespace orc::shared;
  G.allocActions().push_back(
      {cantFail(WrapperFunctionCall::Create<SPSArgList<>>(
           SigningFunctionSym.getAddress())),
       {}});

  return Error::success();
}

} // namespace aarch64
} // namespace jitlink
} // namespace llvm
