//===-- WebAssemblySelectionDAGInfo.cpp - WebAssembly SelectionDAG Info ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the WebAssemblySelectionDAGInfo class.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyTargetMachine.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-selectiondag-info"

WebAssemblySelectionDAGInfo::~WebAssemblySelectionDAGInfo() = default; // anchor

SDValue WebAssemblySelectionDAGInfo::EmitTargetCodeForMemcpy(
    SelectionDAG &DAG, const SDLoc &DL, SDValue Chain, SDValue Op1, SDValue Op2,
    SDValue Op3, unsigned Align, bool isVolatile, bool AlwaysInline,
    MachinePointerInfo DstPtrInfo, MachinePointerInfo SrcPtrInfo) const {
  if (!DAG.getMachineFunction()
           .getSubtarget<WebAssemblySubtarget>()
           .hasBulkMemory())
    return SDValue();

  return DAG.getNode(WebAssemblyISD::MEMORY_COPY, DL, MVT::Other, Chain, Op1,
                     Op2, Op3);
}
