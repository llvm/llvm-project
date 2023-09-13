//===------ aarch32.h - Generic JITLink arm/thumb utilities -----*- C++ -*-===//
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

#ifndef LLVM_EXECUTIONENGINE_JITLINK_AARCH32
#define LLVM_EXECUTIONENGINE_JITLINK_AARCH32

#include "TableManager.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/Support/ARMBuildAttributes.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace jitlink {
namespace aarch32 {

/// Check whether the given target flags are set for this Symbol.
bool hasTargetFlags(Symbol &Sym, TargetFlagsType Flags);

/// JITLink-internal AArch32 fixup kinds
enum EdgeKind_aarch32 : Edge::Kind {

  ///
  /// Relocations of class Data respect target endianness (unless otherwise
  /// specified)
  ///
  FirstDataRelocation = Edge::FirstRelocation,

  /// Relative 32-bit value relocation
  Data_Delta32 = FirstDataRelocation,

  /// Absolute 32-bit value relocation
  Data_Pointer32,

  LastDataRelocation = Data_Pointer32,

  ///
  /// Relocations of class Arm (covers fixed-width 4-byte instruction subset)
  ///
  FirstArmRelocation,

  /// Write immediate value for PC-relative branch with link (can bridge between
  /// Arm and Thumb).
  Arm_Call = FirstArmRelocation,

  /// Write immediate value for (unconditional) PC-relative branch without link.
  Arm_Jump24,

  LastArmRelocation = Arm_Jump24,

  ///
  /// Relocations of class Thumb16 and Thumb32 (covers Thumb instruction subset)
  ///
  FirstThumbRelocation,

  /// Write immediate value for PC-relative branch with link (can bridge between
  /// Arm and Thumb).
  Thumb_Call = FirstThumbRelocation,

  /// Write immediate value for (unconditional) PC-relative branch without link.
  Thumb_Jump24,

  /// Write immediate value to the lower halfword of the destination register
  Thumb_MovwAbsNC,

  /// Write immediate value to the top halfword of the destination register
  Thumb_MovtAbs,

  LastThumbRelocation = Thumb_MovtAbs,
};

/// Flags enum for AArch32-specific symbol properties
enum TargetFlags_aarch32 : TargetFlagsType {
  ThumbSymbol = 1 << 0,
};

/// Human-readable name for a given CPU architecture kind
const char *getCPUArchName(ARMBuildAttrs::CPUArch K);

/// Get a human-readable name for the given AArch32 edge kind.
const char *getEdgeKindName(Edge::Kind K);

/// AArch32 uses stubs for a number of purposes, like branch range extension
/// or interworking between Arm and Thumb instruction subsets.
///
/// Stub implementations vary depending on CPU architecture (v4, v6, v7),
/// instruction subset and branch type (absolute/PC-relative).
///
/// For each kind of stub, the StubsFlavor defines one concrete form that is
/// used throughout the LinkGraph.
///
/// Stubs are often called "veneers" in the official docs and online.
///
enum StubsFlavor {
  Unsupported = 0,
  Thumbv7,
};

/// JITLink sub-arch configuration for Arm CPU models
struct ArmConfig {
  bool J1J2BranchEncoding = false;
  StubsFlavor Stubs = Unsupported;
};

/// Obtain the sub-arch configuration for a given Arm CPU model.
inline ArmConfig getArmConfigForCPUArch(ARMBuildAttrs::CPUArch CPUArch) {
  ArmConfig ArmCfg;
  switch (CPUArch) {
  case ARMBuildAttrs::v7:
  case ARMBuildAttrs::v8_A:
    ArmCfg.J1J2BranchEncoding = true;
    ArmCfg.Stubs = Thumbv7;
    break;
  default:
    DEBUG_WITH_TYPE("jitlink", {
      dbgs() << "  Warning: ARM config not defined for CPU architecture "
             << getCPUArchName(CPUArch);
    });
    break;
  }
  return ArmCfg;
}

/// Immutable pair of halfwords, Hi and Lo, with overflow check
struct HalfWords {
  constexpr HalfWords() : Hi(0), Lo(0) {}
  constexpr HalfWords(uint32_t Hi, uint32_t Lo) : Hi(Hi), Lo(Lo) {
    assert(isUInt<16>(Hi) && "Overflow in first half-word");
    assert(isUInt<16>(Lo) && "Overflow in second half-word");
  }
  const uint16_t Hi; // First halfword
  const uint16_t Lo; // Second halfword
};

/// Collection of named constants per fixup kind. It may contain but is not
/// limited to the following entries:
///
///   Opcode      - Values of the op-code bits in the instruction, with
///                 unaffected bits nulled
///   OpcodeMask  - Mask with all bits set that encode the op-code
///   ImmMask     - Mask with all bits set that encode the immediate value
///   RegMask     - Mask with all bits set that encode the register
///
template <EdgeKind_aarch32 Kind> struct FixupInfo {};

template <> struct FixupInfo<Arm_Jump24> {
  static constexpr uint32_t Opcode = 0x0a000000;
  static constexpr uint32_t OpcodeMask = 0x0f000000;
  static constexpr uint32_t ImmMask = 0x00ffffff;
  static constexpr uint32_t Unconditional = 0xe0000000;
  static constexpr uint32_t CondMask = 0xe0000000; // excluding BLX bit
};

template <> struct FixupInfo<Arm_Call> : public FixupInfo<Arm_Jump24> {
  static constexpr uint32_t OpcodeMask = 0x0e000000;
  static constexpr uint32_t BitH = 0x01000000;
  static constexpr uint32_t BitBlx = 0x10000000;
};

template <> struct FixupInfo<Thumb_Jump24> {
  static constexpr HalfWords Opcode{0xf000, 0x8000};
  static constexpr HalfWords OpcodeMask{0xf800, 0x8000};
  static constexpr HalfWords ImmMask{0x07ff, 0x2fff};
};

template <> struct FixupInfo<Thumb_Call> {
  static constexpr HalfWords Opcode{0xf000, 0xc000};
  static constexpr HalfWords OpcodeMask{0xf800, 0xc000};
  static constexpr HalfWords ImmMask{0x07ff, 0x2fff};
  static constexpr uint16_t LoBitH = 0x0001;
  static constexpr uint16_t LoBitNoBlx = 0x1000;
};

template <> struct FixupInfo<Thumb_MovtAbs> {
  static constexpr HalfWords Opcode{0xf2c0, 0x0000};
  static constexpr HalfWords OpcodeMask{0xfbf0, 0x8000};
  static constexpr HalfWords ImmMask{0x040f, 0x70ff};
  static constexpr HalfWords RegMask{0x0000, 0x0f00};
};

template <>
struct FixupInfo<Thumb_MovwAbsNC> : public FixupInfo<Thumb_MovtAbs> {
  static constexpr HalfWords Opcode{0xf240, 0x0000};
};

/// Helper function to read the initial addend for Data-class relocations.
Expected<int64_t> readAddendData(LinkGraph &G, Block &B, const Edge &E);

/// Helper function to read the initial addend for Arm-class relocations.
Expected<int64_t> readAddendArm(LinkGraph &G, Block &B, const Edge &E);

/// Helper function to read the initial addend for Thumb-class relocations.
Expected<int64_t> readAddendThumb(LinkGraph &G, Block &B, const Edge &E,
                                  const ArmConfig &ArmCfg);

/// Read the initial addend for a REL-type relocation. It's the value encoded
/// in the immediate field of the fixup location by the compiler.
inline Expected<int64_t> readAddend(LinkGraph &G, Block &B, const Edge &E,
                                    const ArmConfig &ArmCfg) {
  Edge::Kind Kind = E.getKind();
  if (Kind <= LastDataRelocation)
    return readAddendData(G, B, E);

  if (Kind <= LastArmRelocation)
    return readAddendArm(G, B, E);

  if (Kind <= LastThumbRelocation)
    return readAddendThumb(G, B, E, ArmCfg);

  llvm_unreachable("Relocation must be of class Data, Arm or Thumb");
}

/// Helper function to apply the fixup for Data-class relocations.
Error applyFixupData(LinkGraph &G, Block &B, const Edge &E);

/// Helper function to apply the fixup for Arm-class relocations.
Error applyFixupArm(LinkGraph &G, Block &B, const Edge &E);

/// Helper function to apply the fixup for Thumb-class relocations.
Error applyFixupThumb(LinkGraph &G, Block &B, const Edge &E,
                      const ArmConfig &ArmCfg);

/// Apply fixup expression for edge to block content.
inline Error applyFixup(LinkGraph &G, Block &B, const Edge &E,
                        const ArmConfig &ArmCfg) {
  Edge::Kind Kind = E.getKind();

  if (Kind <= LastDataRelocation)
    return applyFixupData(G, B, E);

  if (Kind <= LastArmRelocation)
    return applyFixupArm(G, B, E);

  if (Kind <= LastThumbRelocation)
    return applyFixupThumb(G, B, E, ArmCfg);

  llvm_unreachable("Relocation must be of class Data, Arm or Thumb");
}

/// Stubs builder for a specific StubsFlavor
///
/// Right now we only have one default stub kind, but we want to extend this
/// and allow creation of specific kinds in the future (e.g. branch range
/// extension or interworking).
///
/// Let's keep it simple for the moment and not wire this through a GOT.
///
template <StubsFlavor Flavor>
class StubsManager : public TableManager<StubsManager<Flavor>> {
public:
  StubsManager() = default;

  /// Name of the object file section that will contain all our stubs.
  static StringRef getSectionName() { return "__llvm_jitlink_STUBS"; }

  /// Implements link-graph traversal via visitExistingEdges().
  bool visitEdge(LinkGraph &G, Block *B, Edge &E) {
    if (E.getTarget().isDefined())
      return false;

    switch (E.getKind()) {
    case Thumb_Call:
    case Thumb_Jump24: {
      DEBUG_WITH_TYPE("jitlink", {
        dbgs() << "  Fixing " << G.getEdgeKindName(E.getKind()) << " edge at "
               << B->getFixupAddress(E) << " (" << B->getAddress() << " + "
               << formatv("{0:x}", E.getOffset()) << ")\n";
      });
      E.setTarget(this->getEntryForTarget(G, E.getTarget()));
      return true;
    }
    }
    return false;
  }

  /// Create a branch range extension stub for the class's flavor.
  Symbol &createEntry(LinkGraph &G, Symbol &Target);

private:
  /// Create a new node in the link-graph for the given stub template.
  template <size_t Size>
  Block &addStub(LinkGraph &G, const uint8_t (&Code)[Size],
                 uint64_t Alignment) {
    ArrayRef<char> Template(reinterpret_cast<const char *>(Code), Size);
    return G.createContentBlock(getStubsSection(G), Template,
                                orc::ExecutorAddr(), Alignment, 0);
  }

  /// Get or create the object file section that will contain all our stubs.
  Section &getStubsSection(LinkGraph &G) {
    if (!StubsSection)
      StubsSection = &G.createSection(getSectionName(),
                                      orc::MemProt::Read | orc::MemProt::Exec);
    return *StubsSection;
  }

  Section *StubsSection = nullptr;
};

/// Create a branch range extension stub with Thumb encoding for v7 CPUs.
template <>
Symbol &StubsManager<Thumbv7>::createEntry(LinkGraph &G, Symbol &Target);

} // namespace aarch32
} // namespace jitlink
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_JITLINK_AARCH32
