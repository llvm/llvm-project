//===-- AGXFrameLowering.cpp - AGX frame lowering -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AGXFrameLowering.h"
#include "AGX.h"
#include "AGXSubtarget.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

unsigned AGXFrameLowering::getCallDepth(const MachineFunction &MF) {
  unsigned Depth = 0;
  if (auto Attr = MF.getFunction().getFnAttribute("agx-call-depth");
      Attr.isStringAttribute())
    if (Attr.getValueAsString().getAsInteger(10, Depth))
      report_fatal_error("invalid AGX call depth");
  return Depth;
}

static cl::opt<bool> MeasuredCallFrames(
    "agx-measured-call-frames", cl::init(false), cl::Hidden,
    cl::desc("Experimental calls: in a module of one kernel, size each call "
             "depth's frame region by the largest frame at that depth (the "
             "kernel then reserves all the stack frame offsets address)"));

// The frame layout of the module's calls, written by its kernel (compiled
// before the functions it calls) and read by those functions:
// !{i64 KernelRegion, i64 DepthRegion}, or, measured, !{i64 KernelRegion,
// i64 0, i64 Depth1, ..., i64 Depth7}, each depth's largest frame recorded as
// the functions are compiled in increasing depth.
static constexpr StringLiteral CallFramesMD = "agx.call.frames";
// Frame offsets are 15-bit immediates.
static constexpr uint64_t AddressableStackBytes = 0x8000;
static constexpr unsigned MaxCallDepth = 7;

static SmallVector<uint64_t, 9> readCallFrames(const Module &M) {
  SmallVector<uint64_t, 9> V;
  if (NamedMDNode *N = M.getNamedMetadata(CallFramesMD))
    if (N->getNumOperands() == 1)
      for (const MDOperand &Op : N->getOperand(0)->operands())
        V.push_back(mdconst::extract<ConstantInt>(Op)->getZExtValue());
  return V;
}

static void writeCallFrames(Module &M, ArrayRef<uint64_t> V) {
  LLVMContext &Ctx = M.getContext();
  Type *I64 = Type::getInt64Ty(Ctx);
  SmallVector<Metadata *, 9> Ops;
  for (uint64_t X : V)
    Ops.push_back(ConstantAsMetadata::get(ConstantInt::get(I64, X)));
  NamedMDNode *N = M.getOrInsertNamedMetadata(CallFramesMD);
  N->clearOperands();
  N->addOperand(MDNode::get(Ctx, Ops));
}

static bool isMeasured(ArrayRef<uint64_t> V) {
  return V.size() == MaxCallDepth + 2 && V[1] == 0;
}

uint64_t AGXFrameLowering::getCalleeFrameOffset(const MachineFunction &MF) {
  unsigned Depth = getCallDepth(MF);
  if (!Depth)
    return 0;
  SmallVector<uint64_t, 9> V = readCallFrames(*MF.getFunction().getParent());
  if (isMeasured(V)) {
    uint64_t Offset = V[0];
    for (unsigned D = 1; D < Depth; ++D)
      Offset += V[D + 1];
    return Offset;
  }
  if (V.size() == 2)
    return V[0] + (Depth - 1) * V[1];
  return Depth * CallFrameRegionBytes;
}

unsigned AGXFrameLowering::getStackAdjustUnits(const MachineFunction &MF) {
  uint64_t Bytes = MF.getFrameInfo().getStackSize();
  if (agxExperimentalLeafCallsEnabled() &&
      MF.getFunction().getCallingConv() == CallingConv::AGX_KERNEL) {
    bool HasCall = false;
    for (const MachineBasicBlock &MBB : MF)
      for (const MachineInstr &MI : MBB)
        HasCall |= MI.getOpcode() == AGX::AGX_CALL;
    if (HasCall) {
      unsigned MaxDepth = 1;
      auto Attr = MF.getFunction().getFnAttribute("agx-max-call-depth");
      if (Attr.isStringAttribute() &&
          Attr.getValueAsString().getAsInteger(10, MaxDepth))
        report_fatal_error("invalid AGX maximum call depth");
      uint64_t Region = CallFrameRegionBytes, Depth = CallFrameRegionBytes;
      Module &M = *const_cast<Module *>(MF.getFunction().getParent());
      unsigned Kernels = count_if(M, [](const Function &F) {
        return !F.isDeclaration() &&
               F.getCallingConv() == CallingConv::AGX_KERNEL;
      });
      if (MeasuredCallFrames && Kernels == 1) {
        // Measured: the kernel's frame, then each depth's largest, which the
        // functions record as they are compiled; all addressable bytes
        // reserved, as their sizes are not known yet.
        uint64_t KernelRegion = alignTo(Bytes, 16);
        if (KernelRegion >= AddressableStackBytes)
          report_fatal_error("AGX stack frame too large");
        SmallVector<uint64_t, 9> V = readCallFrames(M);
        if (!isMeasured(V)) {
          V.assign(MaxCallDepth + 2, 0);
          V[0] = KernelRegion;
          writeCallFrames(M, V);
        }
        Bytes = AddressableStackBytes;
      } else if (Bytes > CallFrameRegionBytes) {
        // A larger kernel region, which the functions it calls must know:
        // only with one kernel in the module (they are compiled after it).
        if (Kernels != 1)
          report_fatal_error(
              "experimental AGX caller frame exceeds 4096 bytes (" +
              MF.getName() + ": " + Twine(Bytes) +
              " bytes) in a module of several kernels");
        Region = alignTo(Bytes, 16);
        if (Region + MaxDepth * Depth > AddressableStackBytes)
          Depth = alignDown((AddressableStackBytes - std::min(Region,
                                                              AddressableStackBytes)) /
                                std::max(MaxDepth, 1u),
                            16);
        if (Depth < 16)
          report_fatal_error("experimental AGX caller frame too large for its "
                             "calls (" + MF.getName() + ": " + Twine(Bytes) +
                             " bytes)");
        writeCallFrames(M, {Region, Depth});
        Bytes = Region + MaxDepth * Depth;
      } else {
        Bytes = Region + MaxDepth * Depth;
      }
    }
  }
  if (!Bytes)
    return 0;
  // Like the vendor compiler, allocate 40 more bytes and round up to 16
  // bytes, in 32-bit units.
  return alignTo(divideCeil(Bytes, 4) + 10, 4);
}

void AGXFrameLowering::emitPrologue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {
  unsigned Units = getStackAdjustUnits(MF);
  if (agxExperimentalLeafCallsEnabled() &&
      MF.getFunction().getCallingConv() != CallingConv::AGX_KERNEL) {
    Module &M = *const_cast<Module *>(MF.getFunction().getParent());
    uint64_t Frame = MF.getFrameInfo().getStackSize();
    unsigned CallDepth = getCallDepth(MF);
    SmallVector<uint64_t, 9> V = readCallFrames(M);
    if (isMeasured(V) && CallDepth) {
      // This depth's largest frame; deeper functions, compiled after every
      // shallower one, are placed after it.
      if (CallDepth > MaxCallDepth)
        report_fatal_error("unsupported AGX nested call depth");
      for (unsigned D = CallDepth + 1; D <= MaxCallDepth; ++D)
        if (V[D + 1])
          report_fatal_error("AGX called functions out of depth order (" +
                             MF.getName() + ")");
      V[CallDepth + 1] = std::max(V[CallDepth + 1], alignTo(Frame, 16));
      writeCallFrames(M, V);
      if (getCalleeFrameOffset(MF) + Frame > AddressableStackBytes)
        report_fatal_error("experimental AGX call frames exceed " +
                           Twine(AddressableStackBytes) + " bytes (" +
                           MF.getName() + ")");
    } else {
      uint64_t DepthBytes = V.size() == 2 ? V[1] : CallFrameRegionBytes;
      if (Frame > DepthBytes)
        report_fatal_error("experimental AGX callee frame exceeds its " +
                           Twine(DepthBytes) + " bytes (" + MF.getName() +
                           ": " + Twine(Frame) + " bytes)");
    }
    // The kernel owns stack_adjust. Save the incoming link only in functions
    // that make another call; a leaf can return directly through r1.
    bool HasCall = false;
    for (const MachineBasicBlock &Block : MF)
      for (const MachineInstr &MI : Block)
        HasCall |= MI.getOpcode() == AGX::AGX_CALL;
    if (HasCall) {
      unsigned Depth = getCallDepth(MF);
      if (Depth == 0 || Depth > 7)
        report_fatal_error("unsupported AGX nested call depth");
      const AGXInstrInfo *TII = MF.getSubtarget<AGXSubtarget>().getInstrInfo();
      BuildMI(MBB, MBB.begin(), DebugLoc(), TII->get(TargetOpcode::COPY),
              AGX::GPR32RegClass.getRegister(29 + Depth))
          .addReg(AGX::R1);
    }
    return;
  }
  if (!Units)
    return;
  if (Units > 0x7fff)
    report_fatal_error("AGX stack frame too large");
  const AGXInstrInfo *TII = MF.getSubtarget<AGXSubtarget>().getInstrInfo();
  BuildMI(MBB, MBB.begin(), DebugLoc(), TII->get(AGX::STACK_ADJUST))
      .addImm(Units);
}

void AGXFrameLowering::emitEpilogue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {
  if (!agxExperimentalLeafCallsEnabled() ||
      MF.getFunction().getCallingConv() == CallingConv::AGX_KERNEL)
    return;
  bool HasCall = false;
  for (const MachineBasicBlock &Block : MF)
    for (const MachineInstr &MI : Block)
      HasCall |= MI.getOpcode() == AGX::AGX_CALL;
  if (!HasCall)
    return;
  unsigned Depth = getCallDepth(MF);
  if (Depth == 0 || Depth > 7)
    report_fatal_error("unsupported AGX nested call depth");
  const AGXInstrInfo *TII = MF.getSubtarget<AGXSubtarget>().getInstrInfo();
  BuildMI(MBB, MBB.getFirstTerminator(), DebugLoc(),
          TII->get(TargetOpcode::COPY), AGX::R1)
      .addReg(AGX::GPR32RegClass.getRegister(29 + Depth));
}
