//=== AArch64CallingConvention.cpp - AArch64 CC impl ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the table-generated and custom routines for the AArch64
// Calling Convention.
//
//===----------------------------------------------------------------------===//

#include "AArch64CallingConvention.h"
#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/IR/CallingConv.h"
using namespace llvm;

static const MCPhysReg XRegList[] = {AArch64::X0, AArch64::X1, AArch64::X2,
                                     AArch64::X3, AArch64::X4, AArch64::X5,
                                     AArch64::X6, AArch64::X7};
static const MCPhysReg HRegList[] = {AArch64::H0, AArch64::H1, AArch64::H2,
                                     AArch64::H3, AArch64::H4, AArch64::H5,
                                     AArch64::H6, AArch64::H7};
static const MCPhysReg SRegList[] = {AArch64::S0, AArch64::S1, AArch64::S2,
                                     AArch64::S3, AArch64::S4, AArch64::S5,
                                     AArch64::S6, AArch64::S7};
static const MCPhysReg DRegList[] = {AArch64::D0, AArch64::D1, AArch64::D2,
                                     AArch64::D3, AArch64::D4, AArch64::D5,
                                     AArch64::D6, AArch64::D7};
static const MCPhysReg QRegList[] = {AArch64::Q0, AArch64::Q1, AArch64::Q2,
                                     AArch64::Q3, AArch64::Q4, AArch64::Q5,
                                     AArch64::Q6, AArch64::Q7};

static bool finishStackBlock(SmallVectorImpl<CCValAssign> &PendingMembers,
                             MVT LocVT, ISD::ArgFlagsTy &ArgFlags,
                             CCState &State, Align SlotAlign) {
  unsigned Size = LocVT.getSizeInBits() / 8;
  const Align StackAlign =
      State.getMachineFunction().getDataLayout().getStackAlignment();
  const Align OrigAlign(ArgFlags.getOrigAlign());
  const Align Alignment = std::min(OrigAlign, StackAlign);

  for (auto &It : PendingMembers) {
    It.convertToMem(State.AllocateStack(Size, std::max(Alignment, SlotAlign)));
    State.addLoc(It);
    SlotAlign = Align(1);
  }

  // All pending members have now been allocated
  PendingMembers.clear();
  return true;
}

/// The Darwin variadic PCS places anonymous arguments in 8-byte stack slots. An
/// [N x Ty] type must still be contiguous in memory though.
static bool CC_AArch64_Custom_Stack_Block(
      unsigned &ValNo, MVT &ValVT, MVT &LocVT, CCValAssign::LocInfo &LocInfo,
      ISD::ArgFlagsTy &ArgFlags, CCState &State) {
  SmallVectorImpl<CCValAssign> &PendingMembers = State.getPendingLocs();

  // Add the argument to the list to be allocated once we know the size of the
  // block.
  PendingMembers.push_back(
      CCValAssign::getPending(ValNo, ValVT, LocVT, LocInfo));

  if (!ArgFlags.isInConsecutiveRegsLast())
    return true;

  return finishStackBlock(PendingMembers, LocVT, ArgFlags, State, Align(8));
}

/// Given an [N x Ty] block, it should be passed in a consecutive sequence of
/// registers. If no such sequence is available, mark the rest of the registers
/// of that type as used and place the argument on the stack.
static bool CC_AArch64_Custom_Block(unsigned &ValNo, MVT &ValVT, MVT &LocVT,
                                    CCValAssign::LocInfo &LocInfo,
                                    ISD::ArgFlagsTy &ArgFlags, CCState &State) {
  const AArch64Subtarget &Subtarget = static_cast<const AArch64Subtarget &>(
      State.getMachineFunction().getSubtarget());
  bool IsDarwinILP32 = Subtarget.isTargetILP32() && Subtarget.isTargetMachO();

  // Try to allocate a contiguous block of registers, each of the correct
  // size to hold one member.
  ArrayRef<MCPhysReg> RegList;
  if (LocVT.SimpleTy == MVT::i64 || (IsDarwinILP32 && LocVT.SimpleTy == MVT::i32))
    RegList = XRegList;
  else if (LocVT.SimpleTy == MVT::f16)
    RegList = HRegList;
  else if (LocVT.SimpleTy == MVT::f32 || LocVT.is32BitVector())
    RegList = SRegList;
  else if (LocVT.SimpleTy == MVT::f64 || LocVT.is64BitVector())
    RegList = DRegList;
  else if (LocVT.SimpleTy == MVT::f128 || LocVT.is128BitVector())
    RegList = QRegList;
  else {
    // Not an array we want to split up after all.
    return false;
  }

  SmallVectorImpl<CCValAssign> &PendingMembers = State.getPendingLocs();

  // Add the argument to the list to be allocated once we know the size of the
  // block.
  PendingMembers.push_back(
      CCValAssign::getPending(ValNo, ValVT, LocVT, LocInfo));

  if (!ArgFlags.isInConsecutiveRegsLast())
    return true;

  // [N x i32] arguments get packed into x-registers on Darwin's arm64_32
  // because that's how the armv7k Clang front-end emits small structs.
  unsigned EltsPerReg = (IsDarwinILP32 && LocVT.SimpleTy == MVT::i32) ? 2 : 1;
  unsigned RegResult = State.AllocateRegBlock(
      RegList, alignTo(PendingMembers.size(), EltsPerReg) / EltsPerReg);
  if (RegResult && EltsPerReg == 1) {
    for (auto &It : PendingMembers) {
      It.convertToReg(RegResult);
      State.addLoc(It);
      ++RegResult;
    }
    PendingMembers.clear();
    return true;
  } else if (RegResult) {
    assert(EltsPerReg == 2 && "unexpected ABI");
    bool UseHigh = false;
    CCValAssign::LocInfo Info;
    for (auto &It : PendingMembers) {
      Info = UseHigh ? CCValAssign::AExtUpper : CCValAssign::ZExt;
      State.addLoc(CCValAssign::getReg(It.getValNo(), MVT::i32, RegResult,
                                       MVT::i64, Info));
      UseHigh = !UseHigh;
      if (!UseHigh)
        ++RegResult;
    }
    PendingMembers.clear();
    return true;
  }

  // Mark all regs in the class as unavailable
  for (auto Reg : RegList)
    State.AllocateReg(Reg);

  const Align SlotAlign = Subtarget.isTargetDarwin() ? Align(1) : Align(8);

  return finishStackBlock(PendingMembers, LocVT, ArgFlags, State, SlotAlign);
}

// TableGen provides definitions of the calling convention analysis entry
// points.
#include "AArch64GenCallingConv.inc"
