//===-- ARMSelectionDAGInfo.h - ARM SelectionDAG Info -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ARM subclass for SelectionDAGTargetInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_ARMSELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_ARM_ARMSELECTIONDAGINFO_H

#include "MCTargetDesc/ARMAddressingModes.h"
#include "llvm/CodeGen/RuntimeLibcallUtil.h"
#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

#define GET_SDNODE_ENUM
#include "ARMGenSDNodeInfo.inc"

namespace llvm {
namespace ARMISD {

enum NodeType : unsigned {
  DYN_ALLOC = GENERATED_OPCODE_END, // Dynamic allocation on the stack.

  MVESEXT,  // Legalization aids for extending a vector into two/four vectors.
  MVEZEXT,  //  or truncating two/four vectors into one. Eventually becomes
  MVETRUNC, //  stack store/load sequence, if not optimized to anything else.

  // Operands of the standard BUILD_VECTOR node are not legalized, which
  // is fine if BUILD_VECTORs are always lowered to shuffles or other
  // operations, but for ARM some BUILD_VECTORs are legal as-is and their
  // operands need to be legalized.  Define an ARM-specific version of
  // BUILD_VECTOR for this purpose.
  BUILD_VECTOR,

  // Vector load N-element structure to all lanes:
  FIRST_MEMORY_OPCODE,
  VLD1DUP = FIRST_MEMORY_OPCODE,
  VLD2DUP,
  VLD3DUP,
  VLD4DUP,

  // NEON loads with post-increment base updates:
  VLD1_UPD,
  VLD2_UPD,
  VLD3_UPD,
  VLD4_UPD,
  VLD2LN_UPD,
  VLD3LN_UPD,
  VLD4LN_UPD,
  VLD1DUP_UPD,
  VLD2DUP_UPD,
  VLD3DUP_UPD,
  VLD4DUP_UPD,
  VLD1x2_UPD,
  VLD1x3_UPD,
  VLD1x4_UPD,

  // NEON stores with post-increment base updates:
  VST1_UPD,
  VST3_UPD,
  VST2LN_UPD,
  VST3LN_UPD,
  VST4LN_UPD,
  VST1x2_UPD,
  VST1x3_UPD,
  VST1x4_UPD,
  LAST_MEMORY_OPCODE = VST1x4_UPD,
};

} // namespace ARMISD

namespace ARM_AM {
  static inline ShiftOpc getShiftOpcForNode(unsigned Opcode) {
    switch (Opcode) {
    default:          return ARM_AM::no_shift;
    case ISD::SHL:    return ARM_AM::lsl;
    case ISD::SRL:    return ARM_AM::lsr;
    case ISD::SRA:    return ARM_AM::asr;
    case ISD::ROTR:   return ARM_AM::ror;
    //case ISD::ROTL:  // Only if imm -> turn into ROTR.
    // Can't handle RRX here, because it would require folding a flag into
    // the addressing mode.  :(  This causes us to miss certain things.
    //case ARMISD::RRX: return ARM_AM::rrx;
    }
  }
}  // end namespace ARM_AM

class ARMSelectionDAGInfo : public SelectionDAGGenTargetInfo {
public:
  ARMSelectionDAGInfo();

  const char *getTargetNodeName(unsigned Opcode) const override;

  bool isTargetMemoryOpcode(unsigned Opcode) const override;

  void verifyTargetNode(const SelectionDAG &DAG,
                        const SDNode *N) const override;

  SDValue EmitTargetCodeForMemcpy(SelectionDAG &DAG, const SDLoc &dl,
                                  SDValue Chain, SDValue Dst, SDValue Src,
                                  SDValue Size, Align Alignment,
                                  bool isVolatile, bool AlwaysInline,
                                  MachinePointerInfo DstPtrInfo,
                                  MachinePointerInfo SrcPtrInfo) const override;

  SDValue
  EmitTargetCodeForMemmove(SelectionDAG &DAG, const SDLoc &dl, SDValue Chain,
                           SDValue Dst, SDValue Src, SDValue Size,
                           Align Alignment, bool isVolatile,
                           MachinePointerInfo DstPtrInfo,
                           MachinePointerInfo SrcPtrInfo) const override;

  // Adjust parameters for memset, see RTABI section 4.3.4
  SDValue EmitTargetCodeForMemset(SelectionDAG &DAG, const SDLoc &dl,
                                  SDValue Chain, SDValue Op1, SDValue Op2,
                                  SDValue Op3, Align Alignment, bool isVolatile,
                                  bool AlwaysInline,
                                  MachinePointerInfo DstPtrInfo) const override;

  SDValue EmitSpecializedLibcall(SelectionDAG &DAG, const SDLoc &dl,
                                 SDValue Chain, SDValue Dst, SDValue Src,
                                 SDValue Size, unsigned Align,
                                 RTLIB::Libcall LC) const;
};

} // namespace llvm

#endif
