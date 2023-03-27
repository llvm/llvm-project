//===--------- aarch32.cpp - Generic JITLink arm/thumb utilities ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic utilities for graphs representing arm/thumb objects.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/aarch32.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {
namespace aarch32 {

/// Encode 22-bit immediate value for branch instructions without J1J2 range
/// extension (formats B T4, BL T1 and BLX T2).
///
///   00000:Imm11H:Imm11L:0 -> [ 00000:Imm11H, 00000:Imm11L ]
///                                            J1^ ^J2 will always be 1
///
HalfWords encodeImmBT4BlT1BlxT2(int64_t Value) {
  constexpr uint32_t J1J2 = 0x2800;
  uint32_t Imm11H = (Value >> 12) & 0x07ff;
  uint32_t Imm11L = (Value >> 1) & 0x07ff;
  return HalfWords{Imm11H, Imm11L | J1J2};
}

/// Decode 22-bit immediate value for branch instructions without J1J2 range
/// extension (formats B T4, BL T1 and BLX T2).
///
///   [ 00000:Imm11H, 00000:Imm11L ] -> 00000:Imm11H:Imm11L:0
///                   J1^ ^J2 will always be 1
///
int64_t decodeImmBT4BlT1BlxT2(uint32_t Hi, uint32_t Lo) {
  uint32_t Imm11H = Hi & 0x07ff;
  uint32_t Imm11L = Lo & 0x07ff;
  return SignExtend64<22>(Imm11H << 12 | Imm11L << 1);
}

/// Encode 25-bit immediate value for branch instructions with J1J2 range
/// extension (formats B T4, BL T1 and BLX T2).
///
///   S:I1:I2:Imm10:Imm11:0 -> [ 00000:S:Imm10, 00:J1:0:J2:Imm11 ]
///
HalfWords encodeImmBT4BlT1BlxT2_J1J2(int64_t Value) {
  uint32_t S = (Value >> 14) & 0x0400;
  uint32_t J1 = (((~(Value >> 10)) ^ (Value >> 11)) & 0x2000);
  uint32_t J2 = (((~(Value >> 11)) ^ (Value >> 13)) & 0x0800);
  uint32_t Imm10 = (Value >> 12) & 0x03ff;
  uint32_t Imm11 = (Value >> 1) & 0x07ff;
  return HalfWords{S | Imm10, J1 | J2 | Imm11};
}

/// Decode 25-bit immediate value for branch instructions with J1J2 range
/// extension (formats B T4, BL T1 and BLX T2).
///
///   [ 00000:S:Imm10, 00:J1:0:J2:Imm11] -> S:I1:I2:Imm10:Imm11:0
///
int64_t decodeImmBT4BlT1BlxT2_J1J2(uint32_t Hi, uint32_t Lo) {
  uint32_t S = Hi & 0x0400;
  uint32_t I1 = ~((Lo ^ (Hi << 3)) << 10) & 0x00800000;
  uint32_t I2 = ~((Lo ^ (Hi << 1)) << 11) & 0x00400000;
  uint32_t Imm10 = Hi & 0x03ff;
  uint32_t Imm11 = Lo & 0x07ff;
  return SignExtend64<25>(S << 14 | I1 | I2 | Imm10 << 12 | Imm11 << 1);
}

/// Encode 16-bit immediate value for move instruction formats MOVT T1 and
/// MOVW T3.
///
///   Imm4:Imm1:Imm3:Imm8 -> [ 00000:i:000000:Imm4, 0:Imm3:0000:Imm8 ]
///
HalfWords encodeImmMovtT1MovwT3(uint16_t Value) {
  uint32_t Imm4 = (Value >> 12) & 0x0f;
  uint32_t Imm1 = (Value >> 11) & 0x01;
  uint32_t Imm3 = (Value >> 8) & 0x07;
  uint32_t Imm8 = Value & 0xff;
  return HalfWords{Imm1 << 10 | Imm4, Imm3 << 12 | Imm8};
}

/// Decode 16-bit immediate value from move instruction formats MOVT T1 and
/// MOVW T3.
///
///   [ 00000:i:000000:Imm4, 0:Imm3:0000:Imm8 ] -> Imm4:Imm1:Imm3:Imm8
///
uint16_t decodeImmMovtT1MovwT3(uint32_t Hi, uint32_t Lo) {
  uint32_t Imm4 = Hi & 0x0f;
  uint32_t Imm1 = (Hi >> 10) & 0x01;
  uint32_t Imm3 = (Lo >> 12) & 0x07;
  uint32_t Imm8 = Lo & 0xff;
  uint32_t Imm16 = Imm4 << 12 | Imm1 << 11 | Imm3 << 8 | Imm8;
  assert(Imm16 <= 0xffff && "Decoded value out-of-range");
  return Imm16;
}

/// Encode register ID for instruction formats MOVT T1 and MOVW T3.
///
///   Rd4 -> [0000000000000000, 0000:Rd4:00000000]
///
HalfWords encodeRegMovtT1MovwT3(int64_t Value) {
  uint32_t Rd4 = (Value & 0x0f) << 8;
  return HalfWords{0, Rd4};
}

/// Decode register ID from instruction formats MOVT T1 and MOVW T3.
///
///   [0000000000000000, 0000:Rd4:00000000] -> Rd4
///
int64_t decodeRegMovtT1MovwT3(uint32_t Hi, uint32_t Lo) {
  uint32_t Rd4 = (Lo >> 8) & 0x0f;
  return Rd4;
}

/// 32-bit Thumb instructions are stored as two little-endian halfwords.
/// An instruction at address A encodes bytes A+1, A in the first halfword (Hi),
/// followed by bytes A+3, A+2 in the second halfword (Lo).
struct WritableThumbRelocation {
  /// Create a writable reference to a Thumb32 fixup.
  WritableThumbRelocation(char *FixupPtr)
      : Hi{*reinterpret_cast<support::ulittle16_t *>(FixupPtr)},
        Lo{*reinterpret_cast<support::ulittle16_t *>(FixupPtr + 2)} {}

  support::ulittle16_t &Hi; // First halfword
  support::ulittle16_t &Lo; // Second halfword
};

struct ThumbRelocation {
  /// Create a read-only reference to a Thumb32 fixup.
  ThumbRelocation(const char *FixupPtr)
      : Hi{*reinterpret_cast<const support::ulittle16_t *>(FixupPtr)},
        Lo{*reinterpret_cast<const support::ulittle16_t *>(FixupPtr + 2)} {}

  /// Create a read-only Thumb32 fixup from a writeable one.
  ThumbRelocation(WritableThumbRelocation &Writable)
      : Hi{Writable.Hi}, Lo(Writable.Lo) {}

  const support::ulittle16_t &Hi; // First halfword
  const support::ulittle16_t &Lo; // Second halfword
};

Error makeUnexpectedOpcodeError(const LinkGraph &G, const ThumbRelocation &R,
                                Edge::Kind Kind) {
  return make_error<JITLinkError>(
      formatv("Invalid opcode [ 0x{0:x4}, 0x{1:x4} ] for relocation: {2}",
              static_cast<uint16_t>(R.Hi), static_cast<uint16_t>(R.Lo),
              G.getEdgeKindName(Kind)));
}

template <EdgeKind_aarch32 Kind> bool checkOpcode(const ThumbRelocation &R) {
  uint16_t Hi = R.Hi & FixupInfo<Kind>::OpcodeMask.Hi;
  uint16_t Lo = R.Lo & FixupInfo<Kind>::OpcodeMask.Lo;
  return Hi == FixupInfo<Kind>::Opcode.Hi && Lo == FixupInfo<Kind>::Opcode.Lo;
}

template <EdgeKind_aarch32 Kind>
bool checkRegister(const ThumbRelocation &R, HalfWords Reg) {
  uint16_t Hi = R.Hi & FixupInfo<Kind>::RegMask.Hi;
  uint16_t Lo = R.Lo & FixupInfo<Kind>::RegMask.Lo;
  return Hi == Reg.Hi && Lo == Reg.Lo;
}

template <EdgeKind_aarch32 Kind>
bool writeRegister(WritableThumbRelocation &R, HalfWords Reg) {
  static constexpr HalfWords Mask = FixupInfo<Kind>::RegMask;
  assert((Mask.Hi & Reg.Hi) == Reg.Hi && (Mask.Hi & Reg.Hi) == Reg.Hi &&
         "Value bits exceed bit range of given mask");
  R.Hi = (R.Hi & ~Mask.Hi) | Reg.Hi;
  R.Lo = (R.Lo & ~Mask.Lo) | Reg.Lo;
}

template <EdgeKind_aarch32 Kind>
void writeImmediate(WritableThumbRelocation &R, HalfWords Imm) {
  static constexpr HalfWords Mask = FixupInfo<Kind>::ImmMask;
  assert((Mask.Hi & Imm.Hi) == Imm.Hi && (Mask.Hi & Imm.Hi) == Imm.Hi &&
         "Value bits exceed bit range of given mask");
  R.Hi = (R.Hi & ~Mask.Hi) | Imm.Hi;
  R.Lo = (R.Lo & ~Mask.Lo) | Imm.Lo;
}

Expected<int64_t> readAddendData(LinkGraph &G, Block &B, const Edge &E) {
  support::endianness Endian = G.getEndianness();
  assert(Endian != support::native && "Declare as little or big explicitly");

  Edge::Kind Kind = E.getKind();
  const char *BlockWorkingMem = B.getContent().data();
  const char *FixupPtr = BlockWorkingMem + E.getOffset();

  switch (Kind) {
  case Data_Delta32:
    return SignExtend64<32>(support::endian::read32(FixupPtr, Endian));
  default:
    return make_error<JITLinkError>(
        "In graph " + G.getName() + ", section " + B.getSection().getName() +
        " can not read implicit addend for aarch32 edge kind " +
        G.getEdgeKindName(E.getKind()));
  }
}

Expected<int64_t> readAddendArm(LinkGraph &G, Block &B, const Edge &E) {
  Edge::Kind Kind = E.getKind();

  switch (Kind) {
  case Arm_Call:
    return make_error<JITLinkError>(
        "Addend extraction for relocation type not yet implemented: " +
        StringRef(G.getEdgeKindName(Kind)));
  default:
    return make_error<JITLinkError>(
        "In graph " + G.getName() + ", section " + B.getSection().getName() +
        " can not read implicit addend for aarch32 edge kind " +
        G.getEdgeKindName(E.getKind()));
  }
}

Expected<int64_t> readAddendThumb(LinkGraph &G, Block &B, const Edge &E,
                                  const ArmConfig &ArmCfg) {
  ThumbRelocation R(B.getContent().data() + E.getOffset());
  Edge::Kind Kind = E.getKind();

  switch (Kind) {
  case Thumb_Call:
    if (!checkOpcode<Thumb_Call>(R))
      return makeUnexpectedOpcodeError(G, R, Kind);
    return LLVM_LIKELY(ArmCfg.J1J2BranchEncoding)
               ? decodeImmBT4BlT1BlxT2_J1J2(R.Hi, R.Lo)
               : decodeImmBT4BlT1BlxT2(R.Hi, R.Lo);

  case Thumb_Jump24:
    if (!checkOpcode<Thumb_Jump24>(R))
      return makeUnexpectedOpcodeError(G, R, Kind);
    if (R.Lo & FixupInfo<Thumb_Jump24>::LoBitConditional)
      return make_error<JITLinkError>("Relocation expects an unconditional "
                                      "B.W branch instruction: " +
                                      StringRef(G.getEdgeKindName(Kind)));
    return LLVM_LIKELY(ArmCfg.J1J2BranchEncoding)
                  ? decodeImmBT4BlT1BlxT2_J1J2(R.Hi, R.Lo)
                  : decodeImmBT4BlT1BlxT2(R.Hi, R.Lo);

  case Thumb_MovwAbsNC:
    if (!checkOpcode<Thumb_MovwAbsNC>(R))
      return makeUnexpectedOpcodeError(G, R, Kind);
    // Initial addend is interpreted as a signed value
    return SignExtend64<16>(decodeImmMovtT1MovwT3(R.Hi, R.Lo));

  case Thumb_MovtAbs:
    if (!checkOpcode<Thumb_MovtAbs>(R))
      return makeUnexpectedOpcodeError(G, R, Kind);
    // Initial addend is interpreted as a signed value
    return SignExtend64<16>(decodeImmMovtT1MovwT3(R.Hi, R.Lo));

  default:
    return make_error<JITLinkError>(
        "In graph " + G.getName() + ", section " + B.getSection().getName() +
        " can not read implicit addend for aarch32 edge kind " +
        G.getEdgeKindName(E.getKind()));
  }
}

Error applyFixupData(LinkGraph &G, Block &B, const Edge &E) {
  using namespace support;

  char *BlockWorkingMem = B.getAlreadyMutableContent().data();
  char *FixupPtr = BlockWorkingMem + E.getOffset();

  auto Write32 = [FixupPtr, Endian = G.getEndianness()](int64_t Value) {
    assert(Endian != native && "Must be explicit: little or big");
    assert(isInt<32>(Value) && "Must be in signed 32-bit range");
    uint32_t Imm = static_cast<int32_t>(Value);
    if (LLVM_LIKELY(Endian == little))
      endian::write32<little>(FixupPtr, Imm);
    else
      endian::write32<big>(FixupPtr, Imm);
  };

  Edge::Kind Kind = E.getKind();
  uint64_t FixupAddress = (B.getAddress() + E.getOffset()).getValue();
  int64_t Addend = E.getAddend();
  Symbol &TargetSymbol = E.getTarget();
  uint64_t TargetAddress = TargetSymbol.getAddress().getValue();
  assert(!TargetSymbol.hasTargetFlags(ThumbSymbol));

  // Regular data relocations have size 4, alignment 1 and write the full 32-bit
  // result to the place; no need for overflow checking. There are three
  // exceptions: R_ARM_ABS8, R_ARM_ABS16, R_ARM_PREL31
  switch (Kind) {
  case Data_Delta32: {
    int64_t Value = TargetAddress - FixupAddress + Addend;
    if (!isInt<32>(Value))
      return makeTargetOutOfRangeError(G, B, E);
    Write32(Value);
    return Error::success();
  }
  default:
    return make_error<JITLinkError>(
        "In graph " + G.getName() + ", section " + B.getSection().getName() +
        " encountered unfixable aarch32 edge kind " +
        G.getEdgeKindName(E.getKind()));
  }
}

Error applyFixupArm(LinkGraph &G, Block &B, const Edge &E) {
  Edge::Kind Kind = E.getKind();

  switch (Kind) {
  case Arm_Call:
    return make_error<JITLinkError>(
        "Fix-up for relocation type not yet implemented: " +
        StringRef(G.getEdgeKindName(Kind)));
  default:
    return make_error<JITLinkError>(
        "In graph " + G.getName() + ", section " + B.getSection().getName() +
        " encountered unfixable aarch32 edge kind " +
        G.getEdgeKindName(E.getKind()));
  }
}

Error applyFixupThumb(LinkGraph &G, Block &B, const Edge &E,
                      const ArmConfig &ArmCfg) {
  WritableThumbRelocation R(B.getAlreadyMutableContent().data() +
                            E.getOffset());

  Edge::Kind Kind = E.getKind();
  uint64_t FixupAddress = (B.getAddress() + E.getOffset()).getValue();
  int64_t Addend = E.getAddend();
  Symbol &TargetSymbol = E.getTarget();
  uint64_t TargetAddress = TargetSymbol.getAddress().getValue();
  if (TargetSymbol.hasTargetFlags(ThumbSymbol))
    TargetAddress |= 0x01;

  switch (Kind) {
  case Thumb_Jump24: {
    if (!checkOpcode<Thumb_Jump24>(R))
      return makeUnexpectedOpcodeError(G, R, Kind);
    if (R.Lo & FixupInfo<Thumb_Jump24>::LoBitConditional)
      return make_error<JITLinkError>("Relocation expects an unconditional "
                                      "B.W branch instruction: " +
                                      StringRef(G.getEdgeKindName(Kind)));
    if (!(TargetSymbol.hasTargetFlags(ThumbSymbol)))
      return make_error<JITLinkError>("Branch relocation needs interworking "
                                      "stub when bridging to ARM: " +
                                      StringRef(G.getEdgeKindName(Kind)));

    int64_t Value = TargetAddress - FixupAddress + Addend;
    if (LLVM_LIKELY(ArmCfg.J1J2BranchEncoding)) {
      if (!isInt<25>(Value))
        return makeTargetOutOfRangeError(G, B, E);
      writeImmediate<Thumb_Jump24>(R, encodeImmBT4BlT1BlxT2_J1J2(Value));
    } else {
      if (!isInt<22>(Value))
        return makeTargetOutOfRangeError(G, B, E);
      writeImmediate<Thumb_Jump24>(R, encodeImmBT4BlT1BlxT2(Value));
    }

    return Error::success();
  }

  case Thumb_Call: {
    if (!checkOpcode<Thumb_Call>(R))
      return makeUnexpectedOpcodeError(G, R, Kind);

    int64_t Value = TargetAddress - FixupAddress + Addend;

    // The call instruction itself is Thumb. The call destination can either be
    // Thumb or Arm. We use BL to stay in Thumb and BLX to change to Arm.
    bool TargetIsArm = !TargetSymbol.hasTargetFlags(ThumbSymbol);
    bool InstrIsBlx = (R.Lo & FixupInfo<Thumb_Call>::LoBitNoBlx) == 0;
    if (TargetIsArm != InstrIsBlx) {
      if (LLVM_LIKELY(TargetIsArm)) {
        // Change opcode BL -> BLX and fix range value (account for 4-byte
        // aligned destination while instruction may only be 2-byte aligned
        // and clear Thumb bit).
        R.Lo = R.Lo & ~FixupInfo<Thumb_Call>::LoBitNoBlx;
        R.Lo = R.Lo & ~FixupInfo<Thumb_Call>::LoBitH;
        Value = alignTo(Value, 4);
      } else {
        // Change opcode BLX -> BL and set Thumb bit
        R.Lo = R.Lo & ~FixupInfo<Thumb_Call>::LoBitNoBlx;
        Value |= 0x01;
      }
    }

    if (LLVM_LIKELY(ArmCfg.J1J2BranchEncoding)) {
      if (!isInt<25>(Value))
        return makeTargetOutOfRangeError(G, B, E);
      writeImmediate<Thumb_Call>(R, encodeImmBT4BlT1BlxT2_J1J2(Value));
    } else {
      if (!isInt<22>(Value))
        return makeTargetOutOfRangeError(G, B, E);
      writeImmediate<Thumb_Call>(R, encodeImmBT4BlT1BlxT2(Value));
    }

    assert(((R.Lo & FixupInfo<Thumb_Call>::LoBitNoBlx) ||
            (R.Lo & FixupInfo<Thumb_Call>::LoBitH) == 0) &&
           "Opcode BLX implies H bit is clear (avoid UB in BLX T2)");
    return Error::success();
  }

  case Thumb_MovwAbsNC: {
    if (!checkOpcode<Thumb_MovwAbsNC>(R))
      return makeUnexpectedOpcodeError(G, R, Kind);
    uint16_t Value = (TargetAddress + Addend) & 0xffff;
    writeImmediate<Thumb_MovwAbsNC>(R, encodeImmMovtT1MovwT3(Value));
    return Error::success();
  }

  case Thumb_MovtAbs: {
    if (!checkOpcode<Thumb_MovtAbs>(R))
      return makeUnexpectedOpcodeError(G, R, Kind);
    uint16_t Value = ((TargetAddress + Addend) >> 16) & 0xffff;
    writeImmediate<Thumb_MovtAbs>(R, encodeImmMovtT1MovwT3(Value));
    return Error::success();
  }

  default:
    return make_error<JITLinkError>(
        "In graph " + G.getName() + ", section " + B.getSection().getName() +
        " encountered unfixable aarch32 edge kind " +
        G.getEdgeKindName(E.getKind()));
  }
}

const uint8_t Thumbv7ABS[] = {
    0x40, 0xf2, 0x00, 0x0c, // movw r12, #0x0000    ; lower 16-bit
    0xc0, 0xf2, 0x00, 0x0c, // movt r12, #0x0000    ; upper 16-bit
    0x60, 0x47              // bx   r12
};

template <>
Symbol &StubsManager<Thumbv7>::createEntry(LinkGraph &G, Symbol &Target) {
  constexpr uint64_t Alignment = 4;
  Block &B = addStub(G, Thumbv7ABS, Alignment);
  LLVM_DEBUG({
    const char *StubPtr = B.getContent().data();
    HalfWords Reg12 = encodeRegMovtT1MovwT3(12);
    assert(checkRegister<Thumb_MovwAbsNC>(StubPtr, Reg12) &&
           checkRegister<Thumb_MovtAbs>(StubPtr + 4, Reg12) &&
           "Linker generated stubs may only corrupt register r12 (IP)");
  });
  B.addEdge(Thumb_MovwAbsNC, 0, Target, 0);
  B.addEdge(Thumb_MovtAbs, 4, Target, 0);
  Symbol &Stub = G.addAnonymousSymbol(B, 0, B.getSize(), true, false);
  Stub.setTargetFlags(ThumbSymbol);
  return Stub;
}

const char *getEdgeKindName(Edge::Kind K) {
#define KIND_NAME_CASE(K)                                                      \
  case K:                                                                      \
    return #K;

  switch (K) {
    KIND_NAME_CASE(Data_Delta32)
    KIND_NAME_CASE(Arm_Call)
    KIND_NAME_CASE(Thumb_Call)
    KIND_NAME_CASE(Thumb_Jump24)
    KIND_NAME_CASE(Thumb_MovwAbsNC)
    KIND_NAME_CASE(Thumb_MovtAbs)
  default:
    return getGenericEdgeKindName(K);
  }
#undef KIND_NAME_CASE
}

const char *getCPUArchName(ARMBuildAttrs::CPUArch K) {
#define CPUARCH_NAME_CASE(K)                                                   \
  case K:                                                                      \
    return #K;

  using namespace ARMBuildAttrs;
  switch (K) {
    CPUARCH_NAME_CASE(Pre_v4)
    CPUARCH_NAME_CASE(v4)
    CPUARCH_NAME_CASE(v4T)
    CPUARCH_NAME_CASE(v5T)
    CPUARCH_NAME_CASE(v5TE)
    CPUARCH_NAME_CASE(v5TEJ)
    CPUARCH_NAME_CASE(v6)
    CPUARCH_NAME_CASE(v6KZ)
    CPUARCH_NAME_CASE(v6T2)
    CPUARCH_NAME_CASE(v6K)
    CPUARCH_NAME_CASE(v7)
    CPUARCH_NAME_CASE(v6_M)
    CPUARCH_NAME_CASE(v6S_M)
    CPUARCH_NAME_CASE(v7E_M)
    CPUARCH_NAME_CASE(v8_A)
    CPUARCH_NAME_CASE(v8_R)
    CPUARCH_NAME_CASE(v8_M_Base)
    CPUARCH_NAME_CASE(v8_M_Main)
    CPUARCH_NAME_CASE(v8_1_M_Main)
    CPUARCH_NAME_CASE(v9_A)
  }
  llvm_unreachable("Missing CPUArch in switch?");
#undef CPUARCH_NAME_CASE
}

} // namespace aarch32
} // namespace jitlink
} // namespace llvm
