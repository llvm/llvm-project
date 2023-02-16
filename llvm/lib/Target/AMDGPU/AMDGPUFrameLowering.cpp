//===----------------------- AMDGPUFrameLowering.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==-----------------------------------------------------------------------===//
//
// Interface to describe a layout of a stack frame on a AMDGPU target machine.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUFrameLowering.h"
#include "GCNSubtarget.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;
AMDGPUFrameLowering::AMDGPUFrameLowering(StackDirection D, Align StackAl,
                                         int LAO, Align TransAl)
    : TargetFrameLowering(D, StackAl, LAO, TransAl) {}

AMDGPUFrameLowering::~AMDGPUFrameLowering() = default;

unsigned AMDGPUFrameLowering::getStackWidth(const MachineFunction &MF) const {
  // XXX: Hardcoding to 1 for now.
  //
  // I think the StackWidth should be stored as metadata associated with the
  // MachineFunction.  This metadata can either be added by a frontend, or
  // calculated by a R600 specific LLVM IR pass.
  //
  // The StackWidth determines how stack objects are laid out in memory.
  // For a vector stack variable, like: int4 stack[2], the data will be stored
  // in the following ways depending on the StackWidth.
  //
  // StackWidth = 1:
  //
  // T0.X = stack[0].x
  // T1.X = stack[0].y
  // T2.X = stack[0].z
  // T3.X = stack[0].w
  // T4.X = stack[1].x
  // T5.X = stack[1].y
  // T6.X = stack[1].z
  // T7.X = stack[1].w
  //
  // StackWidth = 2:
  //
  // T0.X = stack[0].x
  // T0.Y = stack[0].y
  // T1.X = stack[0].z
  // T1.Y = stack[0].w
  // T2.X = stack[1].x
  // T2.Y = stack[1].y
  // T3.X = stack[1].z
  // T3.Y = stack[1].w
  //
  // StackWidth = 4:
  // T0.X = stack[0].x
  // T0.Y = stack[0].y
  // T0.Z = stack[0].z
  // T0.W = stack[0].w
  // T1.X = stack[1].x
  // T1.Y = stack[1].y
  // T1.Z = stack[1].z
  // T1.W = stack[1].w
  return 1;
}

DIExprBuilder::Iterator AMDGPUFrameLowering::insertFrameLocation(
    const MachineFunction &MF, DIExprBuilder &Builder,
    DIExprBuilder::Iterator BI, Type *ResultType) const {
  LLVMContext &Context = MF.getMMI().getModule()->getContext();
  const auto &ST = MF.getSubtarget<GCNSubtarget>();
  unsigned AllocaAddrSpace = MF.getDataLayout().getAllocaAddrSpace();
  Type *IntPtrTy = IntegerType::getIntNTy(
      Context, MF.getTarget().getPointerSizeInBits(AllocaAddrSpace));
  ConstantData *WavefrontSizeLog2 = static_cast<ConstantData *>(
      ConstantInt::get(IntPtrTy, ST.getWavefrontSizeLog2(), false));

  SmallVector<DIOp::Variant> FL = { DIOp::Referrer(IntPtrTy) };
  if (!ST.enableFlatScratch())
    FL.append({ DIOp::Constant(WavefrontSizeLog2), DIOp::Shr() });
  FL.append(
      { DIOp::Reinterpret(PointerType::get(ResultType, AllocaAddrSpace)),
        DIOp::Deref(ResultType) });
  return Builder.insert(BI, FL) + FL.size();
}
