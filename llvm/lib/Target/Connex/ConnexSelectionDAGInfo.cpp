//===-- ConnexSelectionDAGInfo.cpp - Connex SelectionDAG Info -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ConnexSelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#include "ConnexSelectionDAGInfo.h"
#include "ConnexTargetMachine.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/DerivedTypes.h"

// Inspired from ARM/ARMSelectionDAGInfo.cpp

using namespace llvm;

#define DEBUG_TYPE "connex-selectiondag-info"

// Emit, if possible, a specialized version of the given Libcall. Typically this
// means selecting the appropriately aligned version, but we also convert memset
// of 0 into memclr.
SDValue ConnexSelectionDAGInfo::EmitSpecializedLibcall(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Dst, SDValue Src,
    SDValue Size, unsigned Align, RTLIB::Libcall LC) const {

  const ConnexSubtarget &Subtarget =
      DAG.getMachineFunction().getSubtarget<ConnexSubtarget>();
  const ConnexTargetLowering *TLI = Subtarget.getTargetLowering();

  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry(Dst,
                         DAG.getDataLayout().getIntPtrType(*DAG.getContext()));
  Args.push_back(Entry);

  /*
  if (AEABILibcall == AEABI_MEMCLR) {
    Entry.Node = Size;
    Args.push_back(Entry);
  } else if (AEABILibcall == AEABI_MEMSET) {
  */
  // Adjust parameters for memset, EABI uses format (ptr, size, value),
  // GNU library uses (ptr, value, size)
  // See RTABI section 4.3.4
  Entry.Node = Size;
  Args.push_back(Entry);

  // Extend or truncate the argument to be an i32 value for the call.
  if (Src.getValueType().bitsGT(MVT::i32))
    Src = DAG.getNode(ISD::TRUNCATE, dl, MVT::i32, Src);
  else if (Src.getValueType().bitsLT(MVT::i32))
    Src = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i32, Src);

  Entry.Node = Src;
  Entry.Ty = Type::getInt32Ty(*DAG.getContext());
  Entry.IsSExt = false;
  Args.push_back(Entry);
  /*
  } else {
    Entry.Node = Src;
    Args.push_back(Entry);

    Entry.Node = Size;
    Args.push_back(Entry);
  }
  */

  static char const *FunctionNames[4][3] = {
      {"__aeabi_memcpy", "__aeabi_memcpy4", "__aeabi_memcpy8"},
      {"__aeabi_memmove", "__aeabi_memmove4", "__aeabi_memmove8"},
      // { "__aeabi_memset",  "__aeabi_memset4",  "__aeabi_memset8"  },
      {"memset", "memset", "memset"},
      {"__aeabi_memclr", "__aeabi_memclr4", "__aeabi_memclr8"}};
  TargetLowering::CallLoweringInfo CLI(DAG);
  CLI.setDebugLoc(dl)
      .setChain(Chain)
      .setCallee(TLI->getLibcallCallingConv(LC),
                 Type::getVoidTy(*DAG.getContext()),
                 DAG.getExternalSymbol(FunctionNames[2][2],
                                       TLI->getPointerTy(DAG.getDataLayout())),
                 std::move(Args))
      .setDiscardResult();
  std::pair<SDValue, SDValue> CallResult = TLI->LowerCallTo(CLI);

  return CallResult.second;
}

SDValue ConnexSelectionDAGInfo::EmitTargetCodeForMemcpy(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Dst, SDValue Src,
    SDValue Size, Align Alignment, bool isVolatile, bool AlwaysInline,
    MachinePointerInfo DstPtrInfo, MachinePointerInfo SrcPtrInfo) const {
  return EmitSpecializedLibcall(DAG, dl, Chain, Dst, Src, Size,
                                Alignment.value(), RTLIB::MEMCPY);
}

SDValue ConnexSelectionDAGInfo::EmitTargetCodeForMemmove(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Dst, SDValue Src,
    SDValue Size, Align Alignment, bool isVolatile,
    MachinePointerInfo DstPtrInfo, MachinePointerInfo SrcPtrInfo) const {
  return EmitSpecializedLibcall(DAG, dl, Chain, Dst, Src, Size,
                                Alignment.value(), RTLIB::MEMMOVE);
}

SDValue ConnexSelectionDAGInfo::EmitTargetCodeForMemset(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Dst, SDValue Src,
    SDValue Size, Align Alignment, bool isVolatile, bool AlwaysInline,
    MachinePointerInfo DstPtrInfo) const {
  LLVM_DEBUG(
      dbgs() << "Entered ConnexSelectionDAGInfo::EmitTargetCodeForMemset()"
             << "\n");

  return EmitSpecializedLibcall(DAG, dl, Chain, Dst, Src, Size,
                                Alignment.value(), RTLIB::MEMSET);
}
