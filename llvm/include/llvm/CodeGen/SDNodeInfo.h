//==------------------------------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SDNODEINFO_H
#define LLVM_CODEGEN_SDNODEINFO_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringTable.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGenTypes/MachineValueType.h"

namespace llvm {

class SDNode;
class SelectionDAG;

enum SDNP {
  SDNPHasChain,
  SDNPOutGlue,
  SDNPInGlue,
  SDNPOptInGlue,
  SDNPMemOperand,
  SDNPVariadic,
};

enum SDTC : uint8_t {
  SDTCisVT,
  SDTCisPtrTy,
  SDTCisInt,
  SDTCisFP,
  SDTCisVec,
  SDTCisSameAs,
  SDTCisVTSmallerThanOp,
  SDTCisOpSmallerThanOp,
  SDTCisEltOfVec,
  SDTCisSubVecOfVec,
  SDTCVecEltisVT,
  SDTCisSameNumEltsAs,
  SDTCisSameSizeAs,
};

enum SDNF : uint8_t {
  SDNFIsStrictFP,
  SDNFIsMemOperand,
};

struct VTByHwModePair {
  uint8_t Mode;
  MVT::SimpleValueType VT;
};

struct SDTypeConstraint {
  SDTC Kind;
  uint8_t ConstrainedValIdx;
  uint8_t ConstrainingValIdx;
  /// For Kind == SDTCisVT or SDTCVecEltisVT:
  /// - if not using HwMode, NumHwModes == 0 and VT is MVT::SimpleValueType;
  /// - otherwise, VT is offset into VTByHwModeTable and NumHwModes specifies
  ///   the number of entries.
  uint8_t NumHwModes;
  uint16_t VT;
};

using SDNodeTSFlags = uint32_t;

struct SDNodeDesc {
  uint16_t NameOffset;
  uint8_t TSFlags;
  uint8_t Flags;

  bool hasFlag(SDNF Flag) const { return Flags & (1 << Flag); }
};

static_assert(sizeof(SDNodeDesc) == 4,
              "Keep target SDNode descriptions compact");

struct SDNodeVerifyDesc {
  uint16_t NumResults;
  int16_t NumOperands;
  uint32_t Properties;
  unsigned ConstraintOffset;
  unsigned ConstraintCount;

  bool hasProperty(SDNP Property) const { return Properties & (1 << Property); }
};

static_assert(sizeof(SDNodeVerifyDesc) == 16,
              "Keep target SDNode verification descriptions compact");

struct SDNodeVerifyInfo {
  const SDNodeVerifyDesc *Descs;
  const VTByHwModePair *VTByHwModeTable;
  const SDTypeConstraint *Constraints;
};

class SDNodeInfo final {
  unsigned NumOpcodes;
  const SDNodeDesc *Descs;
  StringTable Names;
  const SDNodeVerifyInfo *VerifyInfo;

public:
  constexpr SDNodeInfo(unsigned NumOpcodes, const SDNodeDesc *Descs,
                       StringTable Names, const SDNodeVerifyInfo *VerifyInfo)
      : NumOpcodes(NumOpcodes), Descs(Descs), Names(Names),
        VerifyInfo(VerifyInfo) {}

  /// Returns true if there is a generated description for a node with the given
  /// target-specific opcode.
  bool hasDesc(unsigned Opcode) const {
    assert(Opcode >= ISD::BUILTIN_OP_END && "Expected target-specific opcode");
    return Opcode < ISD::BUILTIN_OP_END + NumOpcodes;
  }

  /// Returns the description of a node with the given opcode.
  const SDNodeDesc &getDesc(unsigned Opcode) const {
    assert(hasDesc(Opcode));
    return Descs[Opcode - ISD::BUILTIN_OP_END];
  }

#ifndef NDEBUG
  /// Returns the verification description of a node with the given opcode.
  const SDNodeVerifyDesc &getVerifyDesc(unsigned Opcode) const {
    assert(hasDesc(Opcode));
    assert(VerifyInfo);
    return VerifyInfo->Descs[Opcode - ISD::BUILTIN_OP_END];
  }

  /// Returns operand constraints for a node with the given opcode.
  ArrayRef<SDTypeConstraint> getConstraints(unsigned Opcode) const {
    const SDNodeVerifyDesc &Desc = getVerifyDesc(Opcode);
    return ArrayRef(&VerifyInfo->Constraints[Desc.ConstraintOffset],
                    Desc.ConstraintCount);
  }

  const VTByHwModePair *getVTByHwModeTable() const {
    assert(VerifyInfo);
    return VerifyInfo->VTByHwModeTable;
  }
#endif

  /// Returns the name of the given target-specific opcode, suitable for
  /// debug printing.
  StringRef getName(unsigned Opcode) const {
    return Names[getDesc(Opcode).NameOffset];
  }

  LLVM_ABI void verifyNode(const SelectionDAG &DAG, const SDNode *N) const;
};

static_assert(sizeof(SDNodeInfo) == 5 * sizeof(void *),
              "Keep target SDNode information compact");

} // namespace llvm

#endif // LLVM_CODEGEN_SDNODEINFO_H
