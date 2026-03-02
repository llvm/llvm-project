//===-- AMDGPULanePackedABI.cpp - Lane-packed inreg arg ABI ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPULanePackedABI.h"
#include "AMDGPU.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIRegisterInfo.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

static cl::opt<bool> EnableInRegVGPRLanePacking(
    "amdgpu-inreg-vgpr-lane-packing", cl::Hidden,
    cl::desc(
        "Pack overflow inreg args into VGPR lanes using writelane/readlane"),
    cl::init(false));

bool llvm::isInRegVGPRLanePackingEnabled() {
  return EnableInRegVGPRLanePacking;
}

static const MCPhysReg SGPRList[] = {
    AMDGPU::SGPR0,  AMDGPU::SGPR1,  AMDGPU::SGPR2,  AMDGPU::SGPR3,
    AMDGPU::SGPR4,  AMDGPU::SGPR5,  AMDGPU::SGPR6,  AMDGPU::SGPR7,
    AMDGPU::SGPR8,  AMDGPU::SGPR9,  AMDGPU::SGPR10, AMDGPU::SGPR11,
    AMDGPU::SGPR12, AMDGPU::SGPR13, AMDGPU::SGPR14, AMDGPU::SGPR15,
    AMDGPU::SGPR16, AMDGPU::SGPR17, AMDGPU::SGPR18, AMDGPU::SGPR19,
    AMDGPU::SGPR20, AMDGPU::SGPR21, AMDGPU::SGPR22, AMDGPU::SGPR23,
    AMDGPU::SGPR24, AMDGPU::SGPR25, AMDGPU::SGPR26, AMDGPU::SGPR27,
    AMDGPU::SGPR28, AMDGPU::SGPR29};

static const MCPhysReg VGPRList[] = {
    AMDGPU::VGPR0,  AMDGPU::VGPR1,  AMDGPU::VGPR2,  AMDGPU::VGPR3,
    AMDGPU::VGPR4,  AMDGPU::VGPR5,  AMDGPU::VGPR6,  AMDGPU::VGPR7,
    AMDGPU::VGPR8,  AMDGPU::VGPR9,  AMDGPU::VGPR10, AMDGPU::VGPR11,
    AMDGPU::VGPR12, AMDGPU::VGPR13, AMDGPU::VGPR14, AMDGPU::VGPR15,
    AMDGPU::VGPR16, AMDGPU::VGPR17, AMDGPU::VGPR18, AMDGPU::VGPR19,
    AMDGPU::VGPR20, AMDGPU::VGPR21, AMDGPU::VGPR22, AMDGPU::VGPR23,
    AMDGPU::VGPR24, AMDGPU::VGPR25, AMDGPU::VGPR26, AMDGPU::VGPR27,
    AMDGPU::VGPR28, AMDGPU::VGPR29, AMDGPU::VGPR30, AMDGPU::VGPR31};

template <typename ArgT>
void llvm::analyzeArgsWithLanePacking(CCState &State,
                                      const SmallVectorImpl<ArgT> &Args,
                                      bool IsWave32) {
  MCPhysReg PackingVGPR = AMDGPU::NoRegister;
  unsigned NextLane = 0;
  unsigned LanesPerVGPR = IsWave32 ? 32 : 64;

  for (unsigned I = 0, E = Args.size(); I != E; ++I) {
    const ArgT &Arg = Args[I];
    ISD::ArgFlagsTy Flags = Arg.Flags;
    MVT VT = Arg.VT;

    if (Flags.isByVal()) {
      State.HandleByVal(I, VT, VT, CCValAssign::Full, 1, Align(4), Flags);
      continue;
    }

    MVT LocVT = VT;
    CCValAssign::LocInfo LocInfo = CCValAssign::Full;

    if (VT == MVT::i1)
      LocVT = MVT::i32;

    if ((VT == MVT::i8 || VT == MVT::i16) && (Flags.isSExt() || Flags.isZExt()))
      LocVT = MVT::i32;

    auto AssignToStack = [&]() {
      unsigned Offset = State.AllocateStack(4, Align(4));
      State.addLoc(CCValAssign::getMem(I, VT, Offset, LocVT, LocInfo));
    };

    if (Flags.isInReg()) {
      if (MCPhysReg Reg = State.AllocateReg(SGPRList)) {
        State.addLoc(CCValAssign::getReg(I, VT, Reg, LocVT, LocInfo));
        continue;
      }

      if (NextLane >= LanesPerVGPR || PackingVGPR == AMDGPU::NoRegister) {
        PackingVGPR = State.AllocateReg(VGPRList);
        NextLane = 0;
        if (PackingVGPR == AMDGPU::NoRegister) {
          AssignToStack();
          continue;
        }
      }

      State.addLoc(
          CCValAssign::getCustomReg(I, VT, PackingVGPR, LocVT, LocInfo));
      ++NextLane;
      continue;
    }

    if (MCPhysReg Reg = State.AllocateReg(VGPRList)) {
      State.addLoc(CCValAssign::getReg(I, VT, Reg, LocVT, LocInfo));
      continue;
    }

    AssignToStack();
  }
}

template void llvm::analyzeArgsWithLanePacking<ISD::InputArg>(
    CCState &, const SmallVectorImpl<ISD::InputArg> &, bool);
template void llvm::analyzeArgsWithLanePacking<ISD::OutputArg>(
    CCState &, const SmallVectorImpl<ISD::OutputArg> &, bool);

void llvm::packOverflowInRegToVGPRLanes(
    SmallVectorImpl<CCValAssign> &ArgLocs,
    const std::function<bool(unsigned)> &IsInReg, bool IsWave32) {
  unsigned LanesPerVGPR = IsWave32 ? 32 : 64;
  MCPhysReg PackingVGPR = AMDGPU::NoRegister;
  unsigned NextLane = 0;

  for (unsigned I = 0, E = ArgLocs.size(); I != E; ++I) {
    CCValAssign &VA = ArgLocs[I];
    if (!VA.isRegLoc() || !IsInReg(I))
      continue;
    if (!AMDGPU::VGPR_32RegClass.contains(VA.getLocReg()))
      continue;

    if (PackingVGPR == AMDGPU::NoRegister || NextLane >= LanesPerVGPR) {
      PackingVGPR = VA.getLocReg();
      NextLane = 0;
    }

    ArgLocs[I] =
        CCValAssign::getCustomReg(VA.getValNo(), VA.getValVT(), PackingVGPR,
                                  VA.getLocVT(), VA.getLocInfo());
    ++NextLane;
  }
}

unsigned
llvm::getLaneIndexForPackedArg(const SmallVectorImpl<CCValAssign> &Locs,
                               unsigned Idx) {
  assert(Locs[Idx].needsCustom() && Locs[Idx].isRegLoc());
  MCPhysReg VGPR = Locs[Idx].getLocReg();
  unsigned Lane = 0;
  for (unsigned I = 0; I < Idx; ++I) {
    if (Locs[I].needsCustom() && Locs[I].isRegLoc() &&
        Locs[I].getLocReg() == VGPR)
      ++Lane;
  }
  return Lane;
}
