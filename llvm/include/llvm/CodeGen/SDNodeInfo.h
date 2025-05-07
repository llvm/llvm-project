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

enum SDNF {
  SDNFIsStrictFP,
};

struct SDTypeConstraint {
  SDTC Kind;
  uint8_t OpNo;
  uint8_t OtherOpNo;
  MVT::SimpleValueType VT;
};

struct SDNodeDesc {
  uint16_t NumResults;
  int16_t NumOperands;
  uint32_t Properties;
  uint32_t Flags;
  uint32_t TSFlags;
  unsigned NameOffset;
  unsigned ConstraintOffset;
  unsigned ConstraintCount;

  bool hasProperty(SDNP Property) const { return Properties & (1 << Property); }

  bool hasFlag(SDNF Flag) const { return Flags & (1 << Flag); }
};

class SDNodeInfo final {
  unsigned NumOpcodes;
  const SDNodeDesc *Descs;
  StringTable Names;
  const SDTypeConstraint *Constraints;

public:
  constexpr SDNodeInfo(unsigned NumOpcodes, const SDNodeDesc *Descs,
                       StringTable Names, const SDTypeConstraint *Constraints)
      : NumOpcodes(NumOpcodes), Descs(Descs), Names(Names),
        Constraints(Constraints) {}

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

  /// Returns operand constraints for a node with the given opcode.
  ArrayRef<SDTypeConstraint> getConstraints(unsigned Opcode) const {
    const SDNodeDesc &Desc = getDesc(Opcode);
    return ArrayRef(&Constraints[Desc.ConstraintOffset], Desc.ConstraintCount);
  }

  /// Returns the name of the given target-specific opcode, suitable for
  /// debug printing.
  StringRef getName(unsigned Opcode) const {
    return Names[getDesc(Opcode).NameOffset];
  }

  void verifyNode(const SelectionDAG &DAG, const SDNode *N) const;
};

} // namespace llvm

#endif // LLVM_CODEGEN_SDNODEINFO_H
