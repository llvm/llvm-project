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

  /// Write immediate value for unconditional PC-relative branch with link.
  /// We patch the instruction opcode to account for an instruction-set state
  /// switch: we use the bl instruction to stay in ARM and the blx instruction
  /// to switch to Thumb.
  Arm_Call = FirstArmRelocation,

  /// Write immediate value for conditional PC-relative branch without link.
  /// If the branch target is not ARM, we are forced to generate an explicit
  /// interworking stub.
  Arm_Jump24,

  /// Write immediate value to the lower halfword of the destination register
  Arm_MovwAbsNC,

  /// Write immediate value to the top halfword of the destination register
  Arm_MovtAbs,

  LastArmRelocation = Arm_MovtAbs,

  ///
  /// Relocations of class Thumb16 and Thumb32 (covers Thumb instruction subset)
  ///
  FirstThumbRelocation,

  /// Write immediate value for unconditional PC-relative branch with link.
  /// We patch the instruction opcode to account for an instruction-set state
  /// switch: we use the bl instruction to stay in Thumb and the blx instruction
  /// to switch to ARM.
  Thumb_Call = FirstThumbRelocation,

  /// Write immediate value for PC-relative branch without link. The instruction
  /// can be made conditional by an IT block. If the branch target is not ARM,
  /// we are forced to generate an explicit interworking stub.
  Thumb_Jump24,

  /// Write immediate value to the lower halfword of the destination register
  Thumb_MovwAbsNC,

  /// Write immediate value to the top halfword of the destination register
  Thumb_MovtAbs,

  /// Write PC-relative immediate value to the lower halfword of the destination
  /// register
  Thumb_MovwPrelNC,

  /// Write PC-relative immediate value to the top halfword of the destination
  /// register
  Thumb_MovtPrel,

  LastThumbRelocation = Thumb_MovtPrel,
  LastRelocation = LastThumbRelocation,
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
enum class StubsFlavor {
  Unsupported = 0,
  v7,
};

/// JITLink sub-arch configuration for Arm CPU models
struct ArmConfig {
  bool J1J2BranchEncoding = false;
  StubsFlavor Stubs = StubsFlavor::Unsupported;
};

/// Obtain the sub-arch configuration for a given Arm CPU model.
inline ArmConfig getArmConfigForCPUArch(ARMBuildAttrs::CPUArch CPUArch) {
  ArmConfig ArmCfg;
  switch (CPUArch) {
  case ARMBuildAttrs::v7:
  case ARMBuildAttrs::v8_A:
    ArmCfg.J1J2BranchEncoding = true;
    ArmCfg.Stubs = StubsFlavor::v7;
    break;
  default:
    DEBUG_WITH_TYPE("jitlink", {
      dbgs() << "  Warning: ARM config not defined for CPU architecture "
             << getCPUArchName(CPUArch) << " (" << CPUArch << ")\n";
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

/// FixupInfo base class is required for dynamic lookups.
struct FixupInfoBase {
  static const FixupInfoBase *getDynFixupInfo(Edge::Kind K);
  virtual ~FixupInfoBase() {}
};

/// FixupInfo checks for Arm edge kinds work on 32-bit words
struct FixupInfoArm : public FixupInfoBase {
  bool (*checkOpcode)(uint32_t Wd) = nullptr;
};

/// FixupInfo check for Thumb32 edge kinds work on a pair of 16-bit halfwords
struct FixupInfoThumb : public FixupInfoBase {
  bool (*checkOpcode)(uint16_t Hi, uint16_t Lo) = nullptr;
};

/// Collection of named constants per fixup kind
///
/// Mandatory entries:
///   Opcode      - Values of the op-code bits in the instruction, with
///                 unaffected bits nulled
///   OpcodeMask  - Mask with all bits set that encode the op-code
///
/// Other common entries:
///   ImmMask     - Mask with all bits set that encode the immediate value
///   RegMask     - Mask with all bits set that encode the register
///
/// Specializations can add further custom fields without restrictions.
///
template <EdgeKind_aarch32 Kind> struct FixupInfo {};

struct FixupInfoArmBranch : public FixupInfoArm {
  static constexpr uint32_t Opcode = 0x0a000000;
  static constexpr uint32_t ImmMask = 0x00ffffff;
};

template <> struct FixupInfo<Arm_Jump24> : public FixupInfoArmBranch {
  static constexpr uint32_t OpcodeMask = 0x0f000000;
};

template <> struct FixupInfo<Arm_Call> : public FixupInfoArmBranch {
  static constexpr uint32_t OpcodeMask = 0x0e000000;
  static constexpr uint32_t CondMask = 0xe0000000; // excluding BLX bit
  static constexpr uint32_t Unconditional = 0xe0000000;
  static constexpr uint32_t BitH = 0x01000000;
  static constexpr uint32_t BitBlx = 0x10000000;
};

struct FixupInfoArmMov : public FixupInfoArm {
  static constexpr uint32_t OpcodeMask = 0x0ff00000;
  static constexpr uint32_t ImmMask = 0x000f0fff;
  static constexpr uint32_t RegMask = 0x0000f000;
};

template <> struct FixupInfo<Arm_MovtAbs> : public FixupInfoArmMov {
  static constexpr uint32_t Opcode = 0x03400000;
};

template <> struct FixupInfo<Arm_MovwAbsNC> : public FixupInfoArmMov {
  static constexpr uint32_t Opcode = 0x03000000;
};

template <> struct FixupInfo<Thumb_Jump24> : public FixupInfoThumb {
  static constexpr HalfWords Opcode{0xf000, 0x9000};
  static constexpr HalfWords OpcodeMask{0xf800, 0x9000};
  static constexpr HalfWords ImmMask{0x07ff, 0x2fff};
};

template <> struct FixupInfo<Thumb_Call> : public FixupInfoThumb {
  static constexpr HalfWords Opcode{0xf000, 0xc000};
  static constexpr HalfWords OpcodeMask{0xf800, 0xc000};
  static constexpr HalfWords ImmMask{0x07ff, 0x2fff};
  static constexpr uint16_t LoBitH = 0x0001;
  static constexpr uint16_t LoBitNoBlx = 0x1000;
};

struct FixupInfoThumbMov : public FixupInfoThumb {
  static constexpr HalfWords OpcodeMask{0xfbf0, 0x8000};
  static constexpr HalfWords ImmMask{0x040f, 0x70ff};
  static constexpr HalfWords RegMask{0x0000, 0x0f00};
};

template <> struct FixupInfo<Thumb_MovtAbs> : public FixupInfoThumbMov {
  static constexpr HalfWords Opcode{0xf2c0, 0x0000};
};

template <> struct FixupInfo<Thumb_MovtPrel> : public FixupInfoThumbMov {
  static constexpr HalfWords Opcode{0xf2c0, 0x0000};
};

template <> struct FixupInfo<Thumb_MovwAbsNC> : public FixupInfoThumbMov {
  static constexpr HalfWords Opcode{0xf240, 0x0000};
};

template <> struct FixupInfo<Thumb_MovwPrelNC> : public FixupInfoThumbMov {
  static constexpr HalfWords Opcode{0xf240, 0x0000};
};

/// Helper function to read the initial addend for Data-class relocations.
Expected<int64_t> readAddendData(LinkGraph &G, Block &B, Edge::OffsetT Offset,
                                 Edge::Kind Kind);

/// Helper function to read the initial addend for Arm-class relocations.
Expected<int64_t> readAddendArm(LinkGraph &G, Block &B, Edge::OffsetT Offset,
                                Edge::Kind Kind);

/// Helper function to read the initial addend for Thumb-class relocations.
Expected<int64_t> readAddendThumb(LinkGraph &G, Block &B, Edge::OffsetT Offset,
                                  Edge::Kind Kind, const ArmConfig &ArmCfg);

/// Read the initial addend for a REL-type relocation. It's the value encoded
/// in the immediate field of the fixup location by the compiler.
inline Expected<int64_t> readAddend(LinkGraph &G, Block &B,
                                    Edge::OffsetT Offset, Edge::Kind Kind,
                                    const ArmConfig &ArmCfg) {
  if (Kind <= LastDataRelocation)
    return readAddendData(G, B, Offset, Kind);

  if (Kind <= LastArmRelocation)
    return readAddendArm(G, B, Offset, Kind);

  if (Kind <= LastThumbRelocation)
    return readAddendThumb(G, B, Offset, Kind, ArmCfg);

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

/// Stubs builder for v7 emits non-position-independent Thumb stubs.
///
/// Right now we only have one default stub kind, but we want to extend this
/// and allow creation of specific kinds in the future (e.g. branch range
/// extension or interworking).
///
/// Let's keep it simple for the moment and not wire this through a GOT.
///
class StubsManager_v7 : public TableManager<StubsManager_v7> {
public:
  StubsManager_v7() = default;

  /// Name of the object file section that will contain all our stubs.
  static StringRef getSectionName() {
    return "__llvm_jitlink_aarch32_STUBS_Thumbv7";
  }

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

  /// Create a branch range extension stub with Thumb encoding for v7 CPUs.
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

} // namespace aarch32
} // namespace jitlink
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_JITLINK_AARCH32
