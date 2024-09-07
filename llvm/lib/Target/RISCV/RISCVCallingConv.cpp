//===-- RISCVCallingConv.cpp - RISC-V Custom CC Routines ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the custom routines for the RISC-V Calling Convention.
//
//===----------------------------------------------------------------------===//

#include "RISCVCallingConv.h"
#include "RISCVSubtarget.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/MCRegister.h"

using namespace llvm;

// Calling Convention Implementation.
// The expectations for frontend ABI lowering vary from target to target.
// Ideally, an LLVM frontend would be able to avoid worrying about many ABI
// details, but this is a longer term goal. For now, we simply try to keep the
// role of the frontend as simple and well-defined as possible. The rules can
// be summarised as:
// * Never split up large scalar arguments. We handle them here.
// * If a hardfloat calling convention is being used, and the struct may be
// passed in a pair of registers (fp+fp, int+fp), and both registers are
// available, then pass as two separate arguments. If either the GPRs or FPRs
// are exhausted, then pass according to the rule below.
// * If a struct could never be passed in registers or directly in a stack
// slot (as it is larger than 2*XLEN and the floating point rules don't
// apply), then pass it using a pointer with the byval attribute.
// * If a struct is less than 2*XLEN, then coerce to either a two-element
// word-sized array or a 2*XLEN scalar (depending on alignment).
// * The frontend can determine whether a struct is returned by reference or
// not based on its size and fields. If it will be returned by reference, the
// frontend must modify the prototype so a pointer with the sret annotation is
// passed as the first argument. This is not necessary for large scalar
// returns.
// * Struct return values and varargs should be coerced to structs containing
// register-size fields in the same situations they would be for fixed
// arguments.

static const MCPhysReg ArgFPR16s[] = {RISCV::F10_H, RISCV::F11_H, RISCV::F12_H,
                                      RISCV::F13_H, RISCV::F14_H, RISCV::F15_H,
                                      RISCV::F16_H, RISCV::F17_H};
static const MCPhysReg ArgFPR32s[] = {RISCV::F10_F, RISCV::F11_F, RISCV::F12_F,
                                      RISCV::F13_F, RISCV::F14_F, RISCV::F15_F,
                                      RISCV::F16_F, RISCV::F17_F};
static const MCPhysReg ArgFPR64s[] = {RISCV::F10_D, RISCV::F11_D, RISCV::F12_D,
                                      RISCV::F13_D, RISCV::F14_D, RISCV::F15_D,
                                      RISCV::F16_D, RISCV::F17_D};
// This is an interim calling convention and it may be changed in the future.
static const MCPhysReg ArgVRs[] = {
    RISCV::V8,  RISCV::V9,  RISCV::V10, RISCV::V11, RISCV::V12, RISCV::V13,
    RISCV::V14, RISCV::V15, RISCV::V16, RISCV::V17, RISCV::V18, RISCV::V19,
    RISCV::V20, RISCV::V21, RISCV::V22, RISCV::V23};
static const MCPhysReg ArgVRM2s[] = {RISCV::V8M2,  RISCV::V10M2, RISCV::V12M2,
                                     RISCV::V14M2, RISCV::V16M2, RISCV::V18M2,
                                     RISCV::V20M2, RISCV::V22M2};
static const MCPhysReg ArgVRM4s[] = {RISCV::V8M4, RISCV::V12M4, RISCV::V16M4,
                                     RISCV::V20M4};
static const MCPhysReg ArgVRM8s[] = {RISCV::V8M8, RISCV::V16M8};
static const MCPhysReg ArgVRN2M1s[] = {
    RISCV::V8_V9,   RISCV::V9_V10,  RISCV::V10_V11, RISCV::V11_V12,
    RISCV::V12_V13, RISCV::V13_V14, RISCV::V14_V15, RISCV::V15_V16,
    RISCV::V16_V17, RISCV::V17_V18, RISCV::V18_V19, RISCV::V19_V20,
    RISCV::V20_V21, RISCV::V21_V22, RISCV::V22_V23};
static const MCPhysReg ArgVRN3M1s[] = {
    RISCV::V8_V9_V10,   RISCV::V9_V10_V11,  RISCV::V10_V11_V12,
    RISCV::V11_V12_V13, RISCV::V12_V13_V14, RISCV::V13_V14_V15,
    RISCV::V14_V15_V16, RISCV::V15_V16_V17, RISCV::V16_V17_V18,
    RISCV::V17_V18_V19, RISCV::V18_V19_V20, RISCV::V19_V20_V21,
    RISCV::V20_V21_V22, RISCV::V21_V22_V23};
static const MCPhysReg ArgVRN4M1s[] = {
    RISCV::V8_V9_V10_V11,   RISCV::V9_V10_V11_V12,  RISCV::V10_V11_V12_V13,
    RISCV::V11_V12_V13_V14, RISCV::V12_V13_V14_V15, RISCV::V13_V14_V15_V16,
    RISCV::V14_V15_V16_V17, RISCV::V15_V16_V17_V18, RISCV::V16_V17_V18_V19,
    RISCV::V17_V18_V19_V20, RISCV::V18_V19_V20_V21, RISCV::V19_V20_V21_V22,
    RISCV::V20_V21_V22_V23};
static const MCPhysReg ArgVRN5M1s[] = {
    RISCV::V8_V9_V10_V11_V12,   RISCV::V9_V10_V11_V12_V13,
    RISCV::V10_V11_V12_V13_V14, RISCV::V11_V12_V13_V14_V15,
    RISCV::V12_V13_V14_V15_V16, RISCV::V13_V14_V15_V16_V17,
    RISCV::V14_V15_V16_V17_V18, RISCV::V15_V16_V17_V18_V19,
    RISCV::V16_V17_V18_V19_V20, RISCV::V17_V18_V19_V20_V21,
    RISCV::V18_V19_V20_V21_V22, RISCV::V19_V20_V21_V22_V23};
static const MCPhysReg ArgVRN6M1s[] = {
    RISCV::V8_V9_V10_V11_V12_V13,   RISCV::V9_V10_V11_V12_V13_V14,
    RISCV::V10_V11_V12_V13_V14_V15, RISCV::V11_V12_V13_V14_V15_V16,
    RISCV::V12_V13_V14_V15_V16_V17, RISCV::V13_V14_V15_V16_V17_V18,
    RISCV::V14_V15_V16_V17_V18_V19, RISCV::V15_V16_V17_V18_V19_V20,
    RISCV::V16_V17_V18_V19_V20_V21, RISCV::V17_V18_V19_V20_V21_V22,
    RISCV::V18_V19_V20_V21_V22_V23};
static const MCPhysReg ArgVRN7M1s[] = {
    RISCV::V8_V9_V10_V11_V12_V13_V14,   RISCV::V9_V10_V11_V12_V13_V14_V15,
    RISCV::V10_V11_V12_V13_V14_V15_V16, RISCV::V11_V12_V13_V14_V15_V16_V17,
    RISCV::V12_V13_V14_V15_V16_V17_V18, RISCV::V13_V14_V15_V16_V17_V18_V19,
    RISCV::V14_V15_V16_V17_V18_V19_V20, RISCV::V15_V16_V17_V18_V19_V20_V21,
    RISCV::V16_V17_V18_V19_V20_V21_V22, RISCV::V17_V18_V19_V20_V21_V22_V23};
static const MCPhysReg ArgVRN8M1s[] = {RISCV::V8_V9_V10_V11_V12_V13_V14_V15,
                                       RISCV::V9_V10_V11_V12_V13_V14_V15_V16,
                                       RISCV::V10_V11_V12_V13_V14_V15_V16_V17,
                                       RISCV::V11_V12_V13_V14_V15_V16_V17_V18,
                                       RISCV::V12_V13_V14_V15_V16_V17_V18_V19,
                                       RISCV::V13_V14_V15_V16_V17_V18_V19_V20,
                                       RISCV::V14_V15_V16_V17_V18_V19_V20_V21,
                                       RISCV::V15_V16_V17_V18_V19_V20_V21_V22,
                                       RISCV::V16_V17_V18_V19_V20_V21_V22_V23};
static const MCPhysReg ArgVRN2M2s[] = {RISCV::V8M2_V10M2,  RISCV::V10M2_V12M2,
                                       RISCV::V12M2_V14M2, RISCV::V14M2_V16M2,
                                       RISCV::V16M2_V18M2, RISCV::V18M2_V20M2,
                                       RISCV::V20M2_V22M2};
static const MCPhysReg ArgVRN3M2s[] = {
    RISCV::V8M2_V10M2_V12M2,  RISCV::V10M2_V12M2_V14M2,
    RISCV::V12M2_V14M2_V16M2, RISCV::V14M2_V16M2_V18M2,
    RISCV::V16M2_V18M2_V20M2, RISCV::V18M2_V20M2_V22M2};
static const MCPhysReg ArgVRN4M2s[] = {
    RISCV::V8M2_V10M2_V12M2_V14M2, RISCV::V10M2_V12M2_V14M2_V16M2,
    RISCV::V12M2_V14M2_V16M2_V18M2, RISCV::V14M2_V16M2_V18M2_V20M2,
    RISCV::V16M2_V18M2_V20M2_V22M2};
static const MCPhysReg ArgVRN2M4s[] = {RISCV::V8M4_V12M4, RISCV::V12M4_V16M4,
                                       RISCV::V16M4_V20M4};

ArrayRef<MCPhysReg> RISCV::getArgGPRs(const RISCVABI::ABI ABI) {
  // The GPRs used for passing arguments in the ILP32* and LP64* ABIs, except
  // the ILP32E ABI.
  static const MCPhysReg ArgIGPRs[] = {RISCV::X10, RISCV::X11, RISCV::X12,
                                       RISCV::X13, RISCV::X14, RISCV::X15,
                                       RISCV::X16, RISCV::X17};
  // The GPRs used for passing arguments in the ILP32E/ILP64E ABI.
  static const MCPhysReg ArgEGPRs[] = {RISCV::X10, RISCV::X11, RISCV::X12,
                                       RISCV::X13, RISCV::X14, RISCV::X15};

  if (ABI == RISCVABI::ABI_ILP32E || ABI == RISCVABI::ABI_LP64E)
    return ArrayRef(ArgEGPRs);

  return ArrayRef(ArgIGPRs);
}

static ArrayRef<MCPhysReg> getFastCCArgGPRs(const RISCVABI::ABI ABI) {
  // The GPRs used for passing arguments in the FastCC, X5 and X6 might be used
  // for save-restore libcall, so we don't use them.
  // Don't use X7 for fastcc, since Zicfilp uses X7 as the label register.
  static const MCPhysReg FastCCIGPRs[] = {
      RISCV::X10, RISCV::X11, RISCV::X12, RISCV::X13, RISCV::X14, RISCV::X15,
      RISCV::X16, RISCV::X17, RISCV::X28, RISCV::X29, RISCV::X30, RISCV::X31};

  // The GPRs used for passing arguments in the FastCC when using ILP32E/ILP64E.
  static const MCPhysReg FastCCEGPRs[] = {RISCV::X10, RISCV::X11, RISCV::X12,
                                          RISCV::X13, RISCV::X14, RISCV::X15};

  if (ABI == RISCVABI::ABI_ILP32E || ABI == RISCVABI::ABI_LP64E)
    return ArrayRef(FastCCEGPRs);

  return ArrayRef(FastCCIGPRs);
}

// Pass a 2*XLEN argument that has been split into two XLEN values through
// registers or the stack as necessary.
static bool CC_RISCVAssign2XLen(unsigned XLen, CCState &State, CCValAssign VA1,
                                ISD::ArgFlagsTy ArgFlags1, unsigned ValNo2,
                                MVT ValVT2, MVT LocVT2,
                                ISD::ArgFlagsTy ArgFlags2, bool EABI) {
  unsigned XLenInBytes = XLen / 8;
  const RISCVSubtarget &STI =
      State.getMachineFunction().getSubtarget<RISCVSubtarget>();
  ArrayRef<MCPhysReg> ArgGPRs = RISCV::getArgGPRs(STI.getTargetABI());

  if (MCRegister Reg = State.AllocateReg(ArgGPRs)) {
    // At least one half can be passed via register.
    State.addLoc(CCValAssign::getReg(VA1.getValNo(), VA1.getValVT(), Reg,
                                     VA1.getLocVT(), CCValAssign::Full));
  } else {
    // Both halves must be passed on the stack, with proper alignment.
    // TODO: To be compatible with GCC's behaviors, we force them to have 4-byte
    // alignment. This behavior may be changed when RV32E/ILP32E is ratified.
    Align StackAlign(XLenInBytes);
    if (!EABI || XLen != 32)
      StackAlign = std::max(StackAlign, ArgFlags1.getNonZeroOrigAlign());
    State.addLoc(
        CCValAssign::getMem(VA1.getValNo(), VA1.getValVT(),
                            State.AllocateStack(XLenInBytes, StackAlign),
                            VA1.getLocVT(), CCValAssign::Full));
    State.addLoc(CCValAssign::getMem(
        ValNo2, ValVT2, State.AllocateStack(XLenInBytes, Align(XLenInBytes)),
        LocVT2, CCValAssign::Full));
    return false;
  }

  if (MCRegister Reg = State.AllocateReg(ArgGPRs)) {
    // The second half can also be passed via register.
    State.addLoc(
        CCValAssign::getReg(ValNo2, ValVT2, Reg, LocVT2, CCValAssign::Full));
  } else {
    // The second half is passed via the stack, without additional alignment.
    State.addLoc(CCValAssign::getMem(
        ValNo2, ValVT2, State.AllocateStack(XLenInBytes, Align(XLenInBytes)),
        LocVT2, CCValAssign::Full));
  }

  return false;
}

static MCRegister allocateRVVReg(MVT ValVT, unsigned ValNo, CCState &State,
                                 const RISCVTargetLowering &TLI) {
  const TargetRegisterClass *RC = TLI.getRegClassFor(ValVT);
  if (RC == &RISCV::VRRegClass) {
    // Assign the first mask argument to V0.
    // This is an interim calling convention and it may be changed in the
    // future.
    if (ValVT.getVectorElementType() == MVT::i1)
      if (MCRegister Reg = State.AllocateReg(RISCV::V0))
        return Reg;
    return State.AllocateReg(ArgVRs);
  }
  if (RC == &RISCV::VRM2RegClass)
    return State.AllocateReg(ArgVRM2s);
  if (RC == &RISCV::VRM4RegClass)
    return State.AllocateReg(ArgVRM4s);
  if (RC == &RISCV::VRM8RegClass)
    return State.AllocateReg(ArgVRM8s);
  if (RC == &RISCV::VRN2M1RegClass)
    return State.AllocateReg(ArgVRN2M1s);
  if (RC == &RISCV::VRN3M1RegClass)
    return State.AllocateReg(ArgVRN3M1s);
  if (RC == &RISCV::VRN4M1RegClass)
    return State.AllocateReg(ArgVRN4M1s);
  if (RC == &RISCV::VRN5M1RegClass)
    return State.AllocateReg(ArgVRN5M1s);
  if (RC == &RISCV::VRN6M1RegClass)
    return State.AllocateReg(ArgVRN6M1s);
  if (RC == &RISCV::VRN7M1RegClass)
    return State.AllocateReg(ArgVRN7M1s);
  if (RC == &RISCV::VRN8M1RegClass)
    return State.AllocateReg(ArgVRN8M1s);
  if (RC == &RISCV::VRN2M2RegClass)
    return State.AllocateReg(ArgVRN2M2s);
  if (RC == &RISCV::VRN3M2RegClass)
    return State.AllocateReg(ArgVRN3M2s);
  if (RC == &RISCV::VRN4M2RegClass)
    return State.AllocateReg(ArgVRN4M2s);
  if (RC == &RISCV::VRN2M4RegClass)
    return State.AllocateReg(ArgVRN2M4s);
  llvm_unreachable("Unhandled register class for ValueType");
}

// Implements the RISC-V calling convention. Returns true upon failure.
bool llvm::CC_RISCV(unsigned ValNo, MVT ValVT, MVT LocVT,
                    CCValAssign::LocInfo LocInfo, ISD::ArgFlagsTy ArgFlags,
                    CCState &State, bool IsFixed, bool IsRet, Type *OrigTy) {
  const MachineFunction &MF = State.getMachineFunction();
  const DataLayout &DL = MF.getDataLayout();
  const RISCVSubtarget &Subtarget = MF.getSubtarget<RISCVSubtarget>();
  const RISCVTargetLowering &TLI = *Subtarget.getTargetLowering();

  unsigned XLen = DL.getLargestLegalIntTypeSizeInBits();
  assert(XLen == 32 || XLen == 64);
  MVT XLenVT = XLen == 32 ? MVT::i32 : MVT::i64;

  // Static chain parameter must not be passed in normal argument registers,
  // so we assign t2 for it as done in GCC's __builtin_call_with_static_chain
  if (ArgFlags.isNest()) {
    if (MCRegister Reg = State.AllocateReg(RISCV::X7)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  // Any return value split in to more than two values can't be returned
  // directly. Vectors are returned via the available vector registers.
  if (!LocVT.isVector() && IsRet && ValNo > 1)
    return true;

  // UseGPRForF16_F32 if targeting one of the soft-float ABIs, if passing a
  // variadic argument, or if no F16/F32 argument registers are available.
  bool UseGPRForF16_F32 = true;
  // UseGPRForF64 if targeting soft-float ABIs or an FLEN=32 ABI, if passing a
  // variadic argument, or if no F64 argument registers are available.
  bool UseGPRForF64 = true;

  RISCVABI::ABI ABI = Subtarget.getTargetABI();
  switch (ABI) {
  default:
    llvm_unreachable("Unexpected ABI");
  case RISCVABI::ABI_ILP32:
  case RISCVABI::ABI_ILP32E:
  case RISCVABI::ABI_LP64:
  case RISCVABI::ABI_LP64E:
    break;
  case RISCVABI::ABI_ILP32F:
  case RISCVABI::ABI_LP64F:
    UseGPRForF16_F32 = !IsFixed;
    break;
  case RISCVABI::ABI_ILP32D:
  case RISCVABI::ABI_LP64D:
    UseGPRForF16_F32 = !IsFixed;
    UseGPRForF64 = !IsFixed;
    break;
  }

  // FPR16, FPR32, and FPR64 alias each other.
  if (State.getFirstUnallocated(ArgFPR32s) == std::size(ArgFPR32s)) {
    UseGPRForF16_F32 = true;
    UseGPRForF64 = true;
  }

  // From this point on, rely on UseGPRForF16_F32, UseGPRForF64 and
  // similar local variables rather than directly checking against the target
  // ABI.

  ArrayRef<MCPhysReg> ArgGPRs = RISCV::getArgGPRs(ABI);

  if ((ValVT == MVT::f32 && XLen == 32 && Subtarget.hasStdExtZfinx()) ||
      (ValVT == MVT::f64 && XLen == 64 && Subtarget.hasStdExtZdinx())) {
    if (MCRegister Reg = State.AllocateReg(ArgGPRs)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (UseGPRForF16_F32 && (ValVT == MVT::f16 || ValVT == MVT::bf16 ||
                           (ValVT == MVT::f32 && XLen == 64))) {
    MCRegister Reg = State.AllocateReg(ArgGPRs);
    if (Reg) {
      LocVT = XLenVT;
      State.addLoc(
          CCValAssign::getCustomReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (UseGPRForF16_F32 &&
      (ValVT == MVT::f16 || ValVT == MVT::bf16 || ValVT == MVT::f32)) {
    LocVT = XLenVT;
    LocInfo = CCValAssign::BCvt;
  } else if (UseGPRForF64 && XLen == 64 && ValVT == MVT::f64) {
    LocVT = MVT::i64;
    LocInfo = CCValAssign::BCvt;
  }

  // If this is a variadic argument, the RISC-V calling convention requires
  // that it is assigned an 'even' or 'aligned' register if it has 8-byte
  // alignment (RV32) or 16-byte alignment (RV64). An aligned register should
  // be used regardless of whether the original argument was split during
  // legalisation or not. The argument will not be passed by registers if the
  // original type is larger than 2*XLEN, so the register alignment rule does
  // not apply.
  // TODO: To be compatible with GCC's behaviors, we don't align registers
  // currently if we are using ILP32E calling convention. This behavior may be
  // changed when RV32E/ILP32E is ratified.
  unsigned TwoXLenInBytes = (2 * XLen) / 8;
  if (!IsFixed && ArgFlags.getNonZeroOrigAlign() == TwoXLenInBytes &&
      DL.getTypeAllocSize(OrigTy) == TwoXLenInBytes &&
      ABI != RISCVABI::ABI_ILP32E) {
    unsigned RegIdx = State.getFirstUnallocated(ArgGPRs);
    // Skip 'odd' register if necessary.
    if (RegIdx != std::size(ArgGPRs) && RegIdx % 2 == 1)
      State.AllocateReg(ArgGPRs);
  }

  SmallVectorImpl<CCValAssign> &PendingLocs = State.getPendingLocs();
  SmallVectorImpl<ISD::ArgFlagsTy> &PendingArgFlags =
      State.getPendingArgFlags();

  assert(PendingLocs.size() == PendingArgFlags.size() &&
         "PendingLocs and PendingArgFlags out of sync");

  // Handle passing f64 on RV32D with a soft float ABI or when floating point
  // registers are exhausted.
  if (UseGPRForF64 && XLen == 32 && ValVT == MVT::f64) {
    assert(PendingLocs.empty() && "Can't lower f64 if it is split");
    // Depending on available argument GPRS, f64 may be passed in a pair of
    // GPRs, split between a GPR and the stack, or passed completely on the
    // stack. LowerCall/LowerFormalArguments/LowerReturn must recognise these
    // cases.
    MCRegister Reg = State.AllocateReg(ArgGPRs);
    if (!Reg) {
      unsigned StackOffset = State.AllocateStack(8, Align(8));
      State.addLoc(
          CCValAssign::getMem(ValNo, ValVT, StackOffset, LocVT, LocInfo));
      return false;
    }
    LocVT = MVT::i32;
    State.addLoc(CCValAssign::getCustomReg(ValNo, ValVT, Reg, LocVT, LocInfo));
    MCRegister HiReg = State.AllocateReg(ArgGPRs);
    if (HiReg) {
      State.addLoc(
          CCValAssign::getCustomReg(ValNo, ValVT, HiReg, LocVT, LocInfo));
    } else {
      unsigned StackOffset = State.AllocateStack(4, Align(4));
      State.addLoc(
          CCValAssign::getCustomMem(ValNo, ValVT, StackOffset, LocVT, LocInfo));
    }
    return false;
  }

  // Fixed-length vectors are located in the corresponding scalable-vector
  // container types.
  if (ValVT.isFixedLengthVector())
    LocVT = TLI.getContainerForFixedLengthVector(LocVT);

  // Split arguments might be passed indirectly, so keep track of the pending
  // values. Split vectors are passed via a mix of registers and indirectly, so
  // treat them as we would any other argument.
  if (ValVT.isScalarInteger() && (ArgFlags.isSplit() || !PendingLocs.empty())) {
    LocVT = XLenVT;
    LocInfo = CCValAssign::Indirect;
    PendingLocs.push_back(
        CCValAssign::getPending(ValNo, ValVT, LocVT, LocInfo));
    PendingArgFlags.push_back(ArgFlags);
    if (!ArgFlags.isSplitEnd()) {
      return false;
    }
  }

  // If the split argument only had two elements, it should be passed directly
  // in registers or on the stack.
  if (ValVT.isScalarInteger() && ArgFlags.isSplitEnd() &&
      PendingLocs.size() <= 2) {
    assert(PendingLocs.size() == 2 && "Unexpected PendingLocs.size()");
    // Apply the normal calling convention rules to the first half of the
    // split argument.
    CCValAssign VA = PendingLocs[0];
    ISD::ArgFlagsTy AF = PendingArgFlags[0];
    PendingLocs.clear();
    PendingArgFlags.clear();
    return CC_RISCVAssign2XLen(
        XLen, State, VA, AF, ValNo, ValVT, LocVT, ArgFlags,
        ABI == RISCVABI::ABI_ILP32E || ABI == RISCVABI::ABI_LP64E);
  }

  // Allocate to a register if possible, or else a stack slot.
  MCRegister Reg;
  unsigned StoreSizeBytes = XLen / 8;
  Align StackAlign = Align(XLen / 8);

  if ((ValVT == MVT::f16 || ValVT == MVT::bf16) && !UseGPRForF16_F32)
    Reg = State.AllocateReg(ArgFPR16s);
  else if (ValVT == MVT::f32 && !UseGPRForF16_F32)
    Reg = State.AllocateReg(ArgFPR32s);
  else if (ValVT == MVT::f64 && !UseGPRForF64)
    Reg = State.AllocateReg(ArgFPR64s);
  else if (ValVT.isVector() || ValVT.isRISCVVectorTuple()) {
    Reg = allocateRVVReg(ValVT, ValNo, State, TLI);
    if (!Reg) {
      // For return values, the vector must be passed fully via registers or
      // via the stack.
      // FIXME: The proposed vector ABI only mandates v8-v15 for return values,
      // but we're using all of them.
      if (IsRet)
        return true;
      // Try using a GPR to pass the address
      if ((Reg = State.AllocateReg(ArgGPRs))) {
        LocVT = XLenVT;
        LocInfo = CCValAssign::Indirect;
      } else if (ValVT.isScalableVector()) {
        LocVT = XLenVT;
        LocInfo = CCValAssign::Indirect;
      } else {
        // Pass fixed-length vectors on the stack.
        LocVT = ValVT;
        StoreSizeBytes = ValVT.getStoreSize();
        // Align vectors to their element sizes, being careful for vXi1
        // vectors.
        StackAlign = MaybeAlign(ValVT.getScalarSizeInBits() / 8).valueOrOne();
      }
    }
  } else {
    Reg = State.AllocateReg(ArgGPRs);
  }

  unsigned StackOffset =
      Reg ? 0 : State.AllocateStack(StoreSizeBytes, StackAlign);

  // If we reach this point and PendingLocs is non-empty, we must be at the
  // end of a split argument that must be passed indirectly.
  if (!PendingLocs.empty()) {
    assert(ArgFlags.isSplitEnd() && "Expected ArgFlags.isSplitEnd()");
    assert(PendingLocs.size() > 2 && "Unexpected PendingLocs.size()");

    for (auto &It : PendingLocs) {
      if (Reg)
        It.convertToReg(Reg);
      else
        It.convertToMem(StackOffset);
      State.addLoc(It);
    }
    PendingLocs.clear();
    PendingArgFlags.clear();
    return false;
  }

  assert((!UseGPRForF16_F32 || !UseGPRForF64 || LocVT == XLenVT ||
          (TLI.getSubtarget().hasVInstructions() &&
           (ValVT.isVector() || ValVT.isRISCVVectorTuple()))) &&
         "Expected an XLenVT or vector types at this stage");

  if (Reg) {
    State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
    return false;
  }

  // When a scalar floating-point value is passed on the stack, no
  // bit-conversion is needed.
  if (ValVT.isFloatingPoint() && LocInfo != CCValAssign::Indirect) {
    assert(!ValVT.isVector());
    LocVT = ValVT;
    LocInfo = CCValAssign::Full;
  }
  State.addLoc(CCValAssign::getMem(ValNo, ValVT, StackOffset, LocVT, LocInfo));
  return false;
}

// FastCC has less than 1% performance improvement for some particular
// benchmark. But theoretically, it may have benefit for some cases.
bool llvm::CC_RISCV_FastCC(unsigned ValNo, MVT ValVT, MVT LocVT,
                           CCValAssign::LocInfo LocInfo,
                           ISD::ArgFlagsTy ArgFlags, CCState &State,
                           bool IsFixed, bool IsRet, Type *OrigTy) {
  const MachineFunction &MF = State.getMachineFunction();
  const RISCVSubtarget &Subtarget = MF.getSubtarget<RISCVSubtarget>();
  const RISCVTargetLowering &TLI = *Subtarget.getTargetLowering();
  RISCVABI::ABI ABI = Subtarget.getTargetABI();

  if (LocVT == MVT::i32 || LocVT == MVT::i64) {
    if (MCRegister Reg = State.AllocateReg(getFastCCArgGPRs(ABI))) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if ((LocVT == MVT::f16 && Subtarget.hasStdExtZfhmin()) ||
      (LocVT == MVT::bf16 && Subtarget.hasStdExtZfbfmin())) {
    static const MCPhysReg FPR16List[] = {
        RISCV::F10_H, RISCV::F11_H, RISCV::F12_H, RISCV::F13_H, RISCV::F14_H,
        RISCV::F15_H, RISCV::F16_H, RISCV::F17_H, RISCV::F0_H,  RISCV::F1_H,
        RISCV::F2_H,  RISCV::F3_H,  RISCV::F4_H,  RISCV::F5_H,  RISCV::F6_H,
        RISCV::F7_H,  RISCV::F28_H, RISCV::F29_H, RISCV::F30_H, RISCV::F31_H};
    if (MCRegister Reg = State.AllocateReg(FPR16List)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::f32 && Subtarget.hasStdExtF()) {
    static const MCPhysReg FPR32List[] = {
        RISCV::F10_F, RISCV::F11_F, RISCV::F12_F, RISCV::F13_F, RISCV::F14_F,
        RISCV::F15_F, RISCV::F16_F, RISCV::F17_F, RISCV::F0_F,  RISCV::F1_F,
        RISCV::F2_F,  RISCV::F3_F,  RISCV::F4_F,  RISCV::F5_F,  RISCV::F6_F,
        RISCV::F7_F,  RISCV::F28_F, RISCV::F29_F, RISCV::F30_F, RISCV::F31_F};
    if (MCRegister Reg = State.AllocateReg(FPR32List)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::f64 && Subtarget.hasStdExtD()) {
    static const MCPhysReg FPR64List[] = {
        RISCV::F10_D, RISCV::F11_D, RISCV::F12_D, RISCV::F13_D, RISCV::F14_D,
        RISCV::F15_D, RISCV::F16_D, RISCV::F17_D, RISCV::F0_D,  RISCV::F1_D,
        RISCV::F2_D,  RISCV::F3_D,  RISCV::F4_D,  RISCV::F5_D,  RISCV::F6_D,
        RISCV::F7_D,  RISCV::F28_D, RISCV::F29_D, RISCV::F30_D, RISCV::F31_D};
    if (MCRegister Reg = State.AllocateReg(FPR64List)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  // Check if there is an available GPR before hitting the stack.
  if ((LocVT == MVT::f16 && Subtarget.hasStdExtZhinxmin()) ||
      (LocVT == MVT::f32 && Subtarget.hasStdExtZfinx()) ||
      (LocVT == MVT::f64 && Subtarget.is64Bit() &&
       Subtarget.hasStdExtZdinx())) {
    if (MCRegister Reg = State.AllocateReg(getFastCCArgGPRs(ABI))) {
      if (LocVT.getSizeInBits() != Subtarget.getXLen()) {
        LocVT = Subtarget.getXLenVT();
        State.addLoc(
            CCValAssign::getCustomReg(ValNo, ValVT, Reg, LocVT, LocInfo));
        return false;
      }
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::f16 || LocVT == MVT::bf16) {
    unsigned Offset2 = State.AllocateStack(2, Align(2));
    State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset2, LocVT, LocInfo));
    return false;
  }

  if (LocVT == MVT::i32 || LocVT == MVT::f32) {
    unsigned Offset4 = State.AllocateStack(4, Align(4));
    State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset4, LocVT, LocInfo));
    return false;
  }

  if (LocVT == MVT::i64 || LocVT == MVT::f64) {
    unsigned Offset5 = State.AllocateStack(8, Align(8));
    State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset5, LocVT, LocInfo));
    return false;
  }

  if (LocVT.isVector()) {
    if (MCRegister Reg = allocateRVVReg(ValVT, ValNo, State, TLI)) {
      // Fixed-length vectors are located in the corresponding scalable-vector
      // container types.
      if (ValVT.isFixedLengthVector())
        LocVT = TLI.getContainerForFixedLengthVector(LocVT);
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }

    // Try and pass the address via a "fast" GPR.
    if (MCRegister GPRReg = State.AllocateReg(getFastCCArgGPRs(ABI))) {
      LocInfo = CCValAssign::Indirect;
      LocVT = Subtarget.getXLenVT();
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, GPRReg, LocVT, LocInfo));
      return false;
    }

    // Pass scalable vectors indirectly by storing the pointer on the stack.
    if (ValVT.isScalableVector()) {
      LocInfo = CCValAssign::Indirect;
      LocVT = Subtarget.getXLenVT();
      unsigned XLen = Subtarget.getXLen();
      unsigned StackOffset = State.AllocateStack(XLen / 8, Align(XLen / 8));
      State.addLoc(
          CCValAssign::getMem(ValNo, ValVT, StackOffset, LocVT, LocInfo));
      return false;
    }

    // Pass fixed-length vectors on the stack.
    auto StackAlign = MaybeAlign(ValVT.getScalarSizeInBits() / 8).valueOrOne();
    unsigned StackOffset =
        State.AllocateStack(ValVT.getStoreSize(), StackAlign);
    State.addLoc(
        CCValAssign::getMem(ValNo, ValVT, StackOffset, LocVT, LocInfo));
    return false;
  }

  return true; // CC didn't match.
}

bool llvm::CC_RISCV_GHC(unsigned ValNo, MVT ValVT, MVT LocVT,
                        CCValAssign::LocInfo LocInfo, ISD::ArgFlagsTy ArgFlags,
                        CCState &State) {
  if (ArgFlags.isNest()) {
    report_fatal_error(
        "Attribute 'nest' is not supported in GHC calling convention");
  }

  static const MCPhysReg GPRList[] = {
      RISCV::X9,  RISCV::X18, RISCV::X19, RISCV::X20, RISCV::X21, RISCV::X22,
      RISCV::X23, RISCV::X24, RISCV::X25, RISCV::X26, RISCV::X27};

  if (LocVT == MVT::i32 || LocVT == MVT::i64) {
    // Pass in STG registers: Base, Sp, Hp, R1, R2, R3, R4, R5, R6, R7, SpLim
    //                        s1    s2  s3  s4  s5  s6  s7  s8  s9  s10 s11
    if (MCRegister Reg = State.AllocateReg(GPRList)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  const RISCVSubtarget &Subtarget =
      State.getMachineFunction().getSubtarget<RISCVSubtarget>();

  if (LocVT == MVT::f32 && Subtarget.hasStdExtF()) {
    // Pass in STG registers: F1, ..., F6
    //                        fs0 ... fs5
    static const MCPhysReg FPR32List[] = {RISCV::F8_F,  RISCV::F9_F,
                                          RISCV::F18_F, RISCV::F19_F,
                                          RISCV::F20_F, RISCV::F21_F};
    if (MCRegister Reg = State.AllocateReg(FPR32List)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::f64 && Subtarget.hasStdExtD()) {
    // Pass in STG registers: D1, ..., D6
    //                        fs6 ... fs11
    static const MCPhysReg FPR64List[] = {RISCV::F22_D, RISCV::F23_D,
                                          RISCV::F24_D, RISCV::F25_D,
                                          RISCV::F26_D, RISCV::F27_D};
    if (MCRegister Reg = State.AllocateReg(FPR64List)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if ((LocVT == MVT::f32 && Subtarget.hasStdExtZfinx()) ||
      (LocVT == MVT::f64 && Subtarget.hasStdExtZdinx() &&
       Subtarget.is64Bit())) {
    if (MCRegister Reg = State.AllocateReg(GPRList)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  report_fatal_error("No registers left in GHC calling convention");
  return true;
}
