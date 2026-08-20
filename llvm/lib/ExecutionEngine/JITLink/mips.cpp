//===-- mips.cpp - Generic JITLink MIPS edge kinds and utilities ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/mips.h"

#include "llvm/ADT/STLExtras.h"

namespace llvm {
namespace jitlink {
namespace mips {

namespace {

constexpr unsigned InstructionSize = sizeof(uint32_t);
constexpr unsigned HalfwordBits = 16;
constexpr unsigned MaxPointerJumpStubInstructions = 9;

enum class Register : uint32_t {
  Zero = 0,
  T9 = 25,
};

enum class Opcode : uint32_t {
  Addiu = 0x09,
  Lui = 0x0f,
  Daddiu = 0x19,
  Lw = 0x23,
  Ld = 0x37,
};

enum class Function : uint32_t {
  Jr = 0x08,
  Jalr = 0x09,
  Dsll = 0x38,
};

constexpr unsigned OpcodeShift = 26;
constexpr unsigned RSShift = 21;
constexpr unsigned RTShift = 16;
constexpr unsigned RDShift = 11;
constexpr unsigned ShiftAmountShift = 6;

constexpr uint32_t encodeIType(Opcode Op, Register RS, Register RT) {
  return (static_cast<uint32_t>(Op) << OpcodeShift) |
         (static_cast<uint32_t>(RS) << RSShift) |
         (static_cast<uint32_t>(RT) << RTShift);
}

constexpr uint32_t encodeRType(Function Fn, Register RS, Register RT,
                               Register RD, unsigned ShiftAmount = 0) {
  return (static_cast<uint32_t>(RS) << RSShift) |
         (static_cast<uint32_t>(RT) << RTShift) |
         (static_cast<uint32_t>(RD) << RDShift) |
         (ShiftAmount << ShiftAmountShift) | static_cast<uint32_t>(Fn);
}

constexpr uint32_t encodeLui(Register RT) {
  return encodeIType(Opcode::Lui, Register::Zero, RT);
}

constexpr uint32_t encodeAddiu(Register RT, Register RS) {
  return encodeIType(Opcode::Addiu, RS, RT);
}

constexpr uint32_t encodeDaddiu(Register RT, Register RS) {
  return encodeIType(Opcode::Daddiu, RS, RT);
}

constexpr uint32_t encodeLoadPointer32(Register RT, Register Base) {
  return encodeIType(Opcode::Lw, Base, RT);
}

constexpr uint32_t encodeLoadPointer64(Register RT, Register Base) {
  return encodeIType(Opcode::Ld, Base, RT);
}

constexpr uint32_t encodeDsll(Register RD, Register RT, unsigned ShiftAmount) {
  return encodeRType(Function::Dsll, Register::Zero, RT, RD, ShiftAmount);
}

constexpr uint32_t encodeIndirectJump(Register Target, bool R6) {
  // JALR with $zero is the release-6 no-link compact jump.
  return encodeRType(R6 ? Function::Jalr : Function::Jr, Target, Register::Zero,
                     Register::Zero);
}

constexpr uint32_t encodeNop() { return 0; }

struct StubFixup {
  Edge::Kind Kind;
  unsigned InstructionIndex;
};

constexpr StubFixup Pointer32StubFixups[] = {{Hi16, 0}, {Lo16, 1}};
constexpr StubFixup Pointer64StubFixups[] = {
    {Highest16, 0}, {Higher16, 1}, {Hi16, 3}, {Lo16, 5}};

} // namespace

const char *getEdgeKindName(Edge::Kind K) {
  switch (K) {
#define KIND_NAME(K)                                                           \
  case K:                                                                      \
    return #K
    KIND_NAME(Pointer32);
    KIND_NAME(Pointer64);
    KIND_NAME(PagePointer32);
    KIND_NAME(PagePointer64);
    KIND_NAME(Delta32);
    KIND_NAME(Delta64);
    KIND_NAME(NegDelta32);
    KIND_NAME(Abs16);
    KIND_NAME(Hi16);
    KIND_NAME(Lo16);
    KIND_NAME(Higher16);
    KIND_NAME(Highest16);
    KIND_NAME(Jump26);
    KIND_NAME(PC16);
    KIND_NAME(PC32);
    KIND_NAME(PC18S3);
    KIND_NAME(PC19S2);
    KIND_NAME(PC21S2);
    KIND_NAME(PC26S2);
    KIND_NAME(PCHi16);
    KIND_NAME(PCLo16);
    KIND_NAME(GPDispHi16);
    KIND_NAME(GPDispLo16);
    KIND_NAME(GPRel16);
    KIND_NAME(GPRel32);
    KIND_NAME(GPRel64);
    KIND_NAME(GOTOffset16);
    KIND_NAME(GOTOffsetHi16);
    KIND_NAME(GOTOffsetLo16);
    KIND_NAME(GOTPageOffset16);
    KIND_NAME(DTPRelHi16);
    KIND_NAME(DTPRelLo16);
    KIND_NAME(DTPRel32);
    KIND_NAME(DTPRel64);
    KIND_NAME(NegGPRelHi16);
    KIND_NAME(NegGPRelLo16);
    KIND_NAME(RequestGOTAndTransformToOffset16);
    KIND_NAME(RequestGOTPageAndTransformToOffset16);
    KIND_NAME(RequestGOTAndTransformToOffsetHi16);
    KIND_NAME(RequestGOTAndTransformToOffsetLo16);
    KIND_NAME(RequestTLSGDAndTransformToOffset16);
    KIND_NAME(RequestTLSLDMAndTransformToOffset16);
#undef KIND_NAME
  default:
    return getGenericEdgeKindName(K);
  }
}

static const char NullPointerContent[sizeof(uint64_t)] = {};

bool isR6(const LinkGraph &G) {
  const auto &Features = G.getFeatures().getFeatures();
  return llvm::is_contained(Features, "+mips32r6") ||
         llvm::is_contained(Features, "+mips64r6");
}

Edge::Kind getPointerEdgeKind(const LinkGraph &G) {
  return G.getPointerSize() == 8 ? Pointer64 : Pointer32;
}

ArrayRef<char> getPointerBlockContent(const LinkGraph &G) {
  return {NullPointerContent, G.getPointerSize()};
}

Symbol &createAnonymousPointer(LinkGraph &G, Section &PointerSection,
                               Symbol *InitialTarget,
                               Edge::AddendT InitialAddend) {
  auto &B = G.createContentBlock(PointerSection, getPointerBlockContent(G),
                                 orc::ExecutorAddr(), G.getPointerSize(), 0);
  if (InitialTarget)
    B.addEdge(getPointerEdgeKind(G), 0, *InitialTarget, InitialAddend);
  return G.addAnonymousSymbol(B, 0, G.getPointerSize(), false, false);
}

Symbol &createAnonymousPointerJumpStub(LinkGraph &G, Section &StubSection,
                                       Symbol &PointerSymbol) {
  // Non-PIC callers do not enter with this graph's $gp.
  unsigned PointerSize = G.getPointerSize();

  SmallVector<uint32_t, MaxPointerJumpStubInstructions> Instructions;
  ArrayRef<StubFixup> Fixups;
  if (PointerSize == 8) {
    Instructions = {
        encodeLui(Register::T9),
        encodeDaddiu(Register::T9, Register::T9),
        encodeDsll(Register::T9, Register::T9, HalfwordBits),
        encodeDaddiu(Register::T9, Register::T9),
        encodeDsll(Register::T9, Register::T9, HalfwordBits),
        encodeDaddiu(Register::T9, Register::T9),
        encodeLoadPointer64(Register::T9, Register::T9),
    };
    Fixups = Pointer64StubFixups;
  } else {
    Instructions = {
        encodeLui(Register::T9),
        encodeAddiu(Register::T9, Register::T9),
        encodeLoadPointer32(Register::T9, Register::T9),
    };
    Fixups = Pointer32StubFixups;
  }
  Instructions.push_back(encodeIndirectJump(Register::T9, isR6(G)));
  Instructions.push_back(encodeNop());

  auto Content = G.allocateBuffer(Instructions.size() * InstructionSize);
  for (auto [Index, Instruction] : llvm::enumerate(Instructions))
    support::endian::write32(Content.data() + Index * InstructionSize,
                             Instruction, G.getEndianness());

  auto &B = G.createContentBlock(StubSection, Content, orc::ExecutorAddr(),
                                 alignof(uint32_t), 0);
  for (const StubFixup &F : Fixups)
    B.addEdge(F.Kind, F.InstructionIndex * InstructionSize, PointerSymbol, 0);
  return G.addAnonymousSymbol(B, 0, Content.size(), true, false);
}

} // namespace mips
} // namespace jitlink
} // namespace llvm
