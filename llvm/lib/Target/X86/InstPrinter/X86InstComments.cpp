//===-- X86InstComments.cpp - Generate verbose-asm comments for instrs ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines functionality used to emit comments about X86 instructions to
// an output stream for -fverbose-asm.
//
//===----------------------------------------------------------------------===//

#include "X86InstComments.h"
#include "MCTargetDesc/X86MCTargetDesc.h"
#include "Utils/X86ShuffleDecode.h"
#include "llvm/MC/MCInst.h"
#include "llvm/CodeGen/MachineValueType.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static unsigned getVectorRegSize(unsigned RegNo) {

  if (X86::ZMM0 <= RegNo && RegNo <= X86::ZMM31)
    return 512;
  if (X86::YMM0 <= RegNo && RegNo <= X86::YMM31)
    return 256;
  if (X86::XMM0 <= RegNo && RegNo <= X86::XMM31)
    return 128;

  llvm_unreachable("Unknown vector reg!");
  return 0;
}

static MVT getRegOperandVectorVT(const MCInst *MI, const MVT &ScalarVT,
                                 unsigned OperandIndex) {
  unsigned OpReg = MI->getOperand(OperandIndex).getReg();
  return MVT::getVectorVT(ScalarVT,
                          getVectorRegSize(OpReg)/ScalarVT.getSizeInBits());
}

/// \brief Extracts the src/dst types for a given zero extension instruction.
/// \note While the number of elements in DstVT type correct, the
/// number in the SrcVT type is expanded to fill the src xmm register and the
/// upper elements may not be included in the dst xmm/ymm register.
static void getZeroExtensionTypes(const MCInst *MI, MVT &SrcVT, MVT &DstVT) {
  switch (MI->getOpcode()) {
  default:
    llvm_unreachable("Unknown zero extension instruction");
  // i8 zero extension
  case X86::PMOVZXBWrm:
  case X86::PMOVZXBWrr:
  case X86::VPMOVZXBWrm:
  case X86::VPMOVZXBWrr:
    SrcVT = MVT::v16i8;
    DstVT = MVT::v8i16;
    break;
  case X86::VPMOVZXBWYrm:
  case X86::VPMOVZXBWYrr:
    SrcVT = MVT::v16i8;
    DstVT = MVT::v16i16;
    break;
  case X86::PMOVZXBDrm:
  case X86::PMOVZXBDrr:
  case X86::VPMOVZXBDrm:
  case X86::VPMOVZXBDrr:
    SrcVT = MVT::v16i8;
    DstVT = MVT::v4i32;
    break;
  case X86::VPMOVZXBDYrm:
  case X86::VPMOVZXBDYrr:
    SrcVT = MVT::v16i8;
    DstVT = MVT::v8i32;
    break;
  case X86::PMOVZXBQrm:
  case X86::PMOVZXBQrr:
  case X86::VPMOVZXBQrm:
  case X86::VPMOVZXBQrr:
    SrcVT = MVT::v16i8;
    DstVT = MVT::v2i64;
    break;
  case X86::VPMOVZXBQYrm:
  case X86::VPMOVZXBQYrr:
    SrcVT = MVT::v16i8;
    DstVT = MVT::v4i64;
    break;
  // i16 zero extension
  case X86::PMOVZXWDrm:
  case X86::PMOVZXWDrr:
  case X86::VPMOVZXWDrm:
  case X86::VPMOVZXWDrr:
    SrcVT = MVT::v8i16;
    DstVT = MVT::v4i32;
    break;
  case X86::VPMOVZXWDYrm:
  case X86::VPMOVZXWDYrr:
    SrcVT = MVT::v8i16;
    DstVT = MVT::v8i32;
    break;
  case X86::PMOVZXWQrm:
  case X86::PMOVZXWQrr:
  case X86::VPMOVZXWQrm:
  case X86::VPMOVZXWQrr:
    SrcVT = MVT::v8i16;
    DstVT = MVT::v2i64;
    break;
  case X86::VPMOVZXWQYrm:
  case X86::VPMOVZXWQYrr:
    SrcVT = MVT::v8i16;
    DstVT = MVT::v4i64;
    break;
  // i32 zero extension
  case X86::PMOVZXDQrm:
  case X86::PMOVZXDQrr:
  case X86::VPMOVZXDQrm:
  case X86::VPMOVZXDQrr:
    SrcVT = MVT::v4i32;
    DstVT = MVT::v2i64;
    break;
  case X86::VPMOVZXDQYrm:
  case X86::VPMOVZXDQYrr:
    SrcVT = MVT::v4i32;
    DstVT = MVT::v4i64;
    break;
  }
}

#define CASE_MASK_INS_COMMON(Inst, Suffix, src)  \
  case X86::V##Inst##Suffix##src:                \
  case X86::V##Inst##Suffix##src##k:             \
  case X86::V##Inst##Suffix##src##kz:

#define CASE_SSE_INS_COMMON(Inst, src)           \
  case X86::Inst##src:

#define CASE_AVX_INS_COMMON(Inst, Suffix, src)  \
  case X86::V##Inst##Suffix##src:

#define CASE_MOVDUP(Inst, src)                  \
  CASE_MASK_INS_COMMON(Inst, Z, r##src)         \
  CASE_MASK_INS_COMMON(Inst, Z256, r##src)      \
  CASE_MASK_INS_COMMON(Inst, Z128, r##src)      \
  CASE_AVX_INS_COMMON(Inst, , r##src)           \
  CASE_AVX_INS_COMMON(Inst, Y, r##src)          \
  CASE_SSE_INS_COMMON(Inst, r##src)             \

#define CASE_VSHUF(Inst, src)                          \
  CASE_MASK_INS_COMMON(SHUFF##Inst, Z, r##src##i)      \
  CASE_MASK_INS_COMMON(SHUFI##Inst, Z, r##src##i)      \
  CASE_MASK_INS_COMMON(SHUFF##Inst, Z256, r##src##i)   \
  CASE_MASK_INS_COMMON(SHUFI##Inst, Z256, r##src##i)   \

/// \brief Extracts the types and if it has memory operand for a given
/// (SHUFF32x4/SHUFF64x2/SHUFI32x4/SHUFI64x2) instruction.
static void getVSHUF64x2FamilyInfo(const MCInst *MI, MVT &VT, bool &HasMemOp) {
  HasMemOp = false;
  switch (MI->getOpcode()) {
  default:
    llvm_unreachable("Unknown VSHUF64x2 family instructions.");
    break;
  CASE_VSHUF(64X2, m)
    HasMemOp = true;        // FALL THROUGH.
  CASE_VSHUF(64X2, r)
    VT = getRegOperandVectorVT(MI, MVT::i64, 0);
    break;
  CASE_VSHUF(32X4, m)
    HasMemOp = true;        // FALL THROUGH.
  CASE_VSHUF(32X4, r)
    VT = getRegOperandVectorVT(MI, MVT::i32, 0);
    break;
  }
}

//===----------------------------------------------------------------------===//
// Top Level Entrypoint
//===----------------------------------------------------------------------===//

/// EmitAnyX86InstComments - This function decodes x86 instructions and prints
/// newline terminated strings to the specified string if desired.  This
/// information is shown in disassembly dumps when verbose assembly is enabled.
bool llvm::EmitAnyX86InstComments(const MCInst *MI, raw_ostream &OS,
                                  const char *(*getRegName)(unsigned)) {
  // If this is a shuffle operation, the switch should fill in this state.
  SmallVector<int, 8> ShuffleMask;
  const char *DestName = nullptr, *Src1Name = nullptr, *Src2Name = nullptr;

  switch (MI->getOpcode()) {
  default:
    // Not an instruction for which we can decode comments.
    return false;

  case X86::BLENDPDrri:
  case X86::VBLENDPDrri:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::BLENDPDrmi:
  case X86::VBLENDPDrmi:
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodeBLENDMask(MVT::v2f64,
                      MI->getOperand(MI->getNumOperands() - 1).getImm(),
                      ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VBLENDPDYrri:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VBLENDPDYrmi:
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodeBLENDMask(MVT::v4f64,
                      MI->getOperand(MI->getNumOperands() - 1).getImm(),
                      ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  case X86::BLENDPSrri:
  case X86::VBLENDPSrri:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::BLENDPSrmi:
  case X86::VBLENDPSrmi:
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodeBLENDMask(MVT::v4f32,
                      MI->getOperand(MI->getNumOperands() - 1).getImm(),
                      ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VBLENDPSYrri:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VBLENDPSYrmi:
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodeBLENDMask(MVT::v8f32,
                      MI->getOperand(MI->getNumOperands() - 1).getImm(),
                      ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  case X86::PBLENDWrri:
  case X86::VPBLENDWrri:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::PBLENDWrmi:
  case X86::VPBLENDWrmi:
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodeBLENDMask(MVT::v8i16,
                      MI->getOperand(MI->getNumOperands() - 1).getImm(),
                      ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VPBLENDWYrri:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VPBLENDWYrmi:
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodeBLENDMask(MVT::v16i16,
                      MI->getOperand(MI->getNumOperands() - 1).getImm(),
                      ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  case X86::VPBLENDDrri:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VPBLENDDrmi:
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodeBLENDMask(MVT::v4i32,
                      MI->getOperand(MI->getNumOperands() - 1).getImm(),
                      ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  case X86::VPBLENDDYrri:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VPBLENDDYrmi:
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodeBLENDMask(MVT::v8i32,
                      MI->getOperand(MI->getNumOperands() - 1).getImm(),
                      ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  case X86::INSERTPSrr:
  case X86::VINSERTPSrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::INSERTPSrm:
  case X86::VINSERTPSrm:
    DestName = getRegName(MI->getOperand(0).getReg());
    Src1Name = getRegName(MI->getOperand(1).getReg());
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodeINSERTPSMask(MI->getOperand(MI->getNumOperands() - 1).getImm(),
                         ShuffleMask);
    break;

  case X86::MOVLHPSrr:
  case X86::VMOVLHPSrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeMOVLHPSMask(2, ShuffleMask);
    break;

  case X86::MOVHLPSrr:
  case X86::VMOVHLPSrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeMOVHLPSMask(2, ShuffleMask);
    break;
  CASE_MOVDUP(MOVSLDUP, r)
    Src1Name = getRegName(MI->getOperand(MI->getNumOperands() - 1).getReg());
    // FALL THROUGH.
  CASE_MOVDUP(MOVSLDUP, m) {
    MVT VT = getRegOperandVectorVT(MI, MVT::f32, 0);
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeMOVSLDUPMask(VT, ShuffleMask);
    break;
  }
  CASE_MOVDUP(MOVSHDUP, r)
    Src1Name = getRegName(MI->getOperand(MI->getNumOperands() - 1).getReg());
    // FALL THROUGH.
  CASE_MOVDUP(MOVSHDUP, m) {
    MVT VT = getRegOperandVectorVT(MI, MVT::f32, 0);
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeMOVSHDUPMask(VT, ShuffleMask);
    break;
  }
  case X86::VMOVDDUPYrr:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    // FALL THROUGH.
  case X86::VMOVDDUPYrm:
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeMOVDDUPMask(MVT::v4f64, ShuffleMask);
    break;

  case X86::MOVDDUPrr:
  case X86::VMOVDDUPrr:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    // FALL THROUGH.
  case X86::MOVDDUPrm:
  case X86::VMOVDDUPrm:
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeMOVDDUPMask(MVT::v2f64, ShuffleMask);
    break;

  case X86::PSLLDQri:
  case X86::VPSLLDQri:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodePSLLDQMask(MVT::v16i8,
                       MI->getOperand(MI->getNumOperands() - 1).getImm(),
                       ShuffleMask);
    break;

  case X86::VPSLLDQYri:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodePSLLDQMask(MVT::v32i8,
                       MI->getOperand(MI->getNumOperands() - 1).getImm(),
                       ShuffleMask);
    break;

  case X86::PSRLDQri:
  case X86::VPSRLDQri:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodePSRLDQMask(MVT::v16i8,
                       MI->getOperand(MI->getNumOperands() - 1).getImm(),
                       ShuffleMask);
    break;

  case X86::VPSRLDQYri:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodePSRLDQMask(MVT::v32i8,
                       MI->getOperand(MI->getNumOperands() - 1).getImm(),
                       ShuffleMask);
    break;

  case X86::PALIGNR128rr:
  case X86::VPALIGNR128rr:
    Src1Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::PALIGNR128rm:
  case X86::VPALIGNR128rm:
    Src2Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodePALIGNRMask(MVT::v16i8,
                        MI->getOperand(MI->getNumOperands() - 1).getImm(),
                        ShuffleMask);
    break;
  case X86::VPALIGNR256rr:
    Src1Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VPALIGNR256rm:
    Src2Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodePALIGNRMask(MVT::v32i8,
                        MI->getOperand(MI->getNumOperands() - 1).getImm(),
                        ShuffleMask);
    break;

  case X86::PSHUFDri:
  case X86::VPSHUFDri:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    // FALL THROUGH.
  case X86::PSHUFDmi:
  case X86::VPSHUFDmi:
    DestName = getRegName(MI->getOperand(0).getReg());
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodePSHUFMask(MVT::v4i32,
                      MI->getOperand(MI->getNumOperands() - 1).getImm(),
                      ShuffleMask);
    break;
  case X86::VPSHUFDYri:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    // FALL THROUGH.
  case X86::VPSHUFDYmi:
    DestName = getRegName(MI->getOperand(0).getReg());
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodePSHUFMask(MVT::v8i32,
                      MI->getOperand(MI->getNumOperands() - 1).getImm(),
                      ShuffleMask);
    break;

  case X86::PSHUFHWri:
  case X86::VPSHUFHWri:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    // FALL THROUGH.
  case X86::PSHUFHWmi:
  case X86::VPSHUFHWmi:
    DestName = getRegName(MI->getOperand(0).getReg());
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodePSHUFHWMask(MVT::v8i16,
                        MI->getOperand(MI->getNumOperands() - 1).getImm(),
                        ShuffleMask);
    break;
  case X86::VPSHUFHWYri:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    // FALL THROUGH.
  case X86::VPSHUFHWYmi:
    DestName = getRegName(MI->getOperand(0).getReg());
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodePSHUFHWMask(MVT::v16i16,
                        MI->getOperand(MI->getNumOperands() - 1).getImm(),
                        ShuffleMask);
    break;
  case X86::PSHUFLWri:
  case X86::VPSHUFLWri:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    // FALL THROUGH.
  case X86::PSHUFLWmi:
  case X86::VPSHUFLWmi:
    DestName = getRegName(MI->getOperand(0).getReg());
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodePSHUFLWMask(MVT::v8i16,
                        MI->getOperand(MI->getNumOperands() - 1).getImm(),
                        ShuffleMask);
    break;
  case X86::VPSHUFLWYri:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    // FALL THROUGH.
  case X86::VPSHUFLWYmi:
    DestName = getRegName(MI->getOperand(0).getReg());
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodePSHUFLWMask(MVT::v16i16,
                        MI->getOperand(MI->getNumOperands() - 1).getImm(),
                        ShuffleMask);
    break;

  case X86::MMX_PSHUFWri:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    // FALL THROUGH.
  case X86::MMX_PSHUFWmi:
    DestName = getRegName(MI->getOperand(0).getReg());
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodePSHUFMask(MVT::v4i16,
                      MI->getOperand(MI->getNumOperands() - 1).getImm(),
                      ShuffleMask);
    break;

  case X86::PSWAPDrr:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    // FALL THROUGH.
  case X86::PSWAPDrm:
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodePSWAPMask(MVT::v2i32, ShuffleMask);
    break;

  case X86::MMX_PUNPCKHBWirr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
  case X86::MMX_PUNPCKHBWirm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKHMask(MVT::v8i8, ShuffleMask);
    break;
  case X86::MMX_PUNPCKHWDirr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
  case X86::MMX_PUNPCKHWDirm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKHMask(MVT::v4i16, ShuffleMask);
    break;
  case X86::MMX_PUNPCKHDQirr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
  case X86::MMX_PUNPCKHDQirm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKHMask(MVT::v2i32, ShuffleMask);
    break;
  case X86::PUNPCKHBWrr:
  case X86::VPUNPCKHBWrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::PUNPCKHBWrm:
  case X86::VPUNPCKHBWrm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKHMask(MVT::v16i8, ShuffleMask);
    break;
  case X86::VPUNPCKHBWYrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VPUNPCKHBWYrm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKHMask(MVT::v32i8, ShuffleMask);
    break;
  case X86::PUNPCKHWDrr:
  case X86::VPUNPCKHWDrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::PUNPCKHWDrm:
  case X86::VPUNPCKHWDrm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKHMask(MVT::v8i16, ShuffleMask);
    break;
  case X86::VPUNPCKHWDYrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VPUNPCKHWDYrm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKHMask(MVT::v16i16, ShuffleMask);
    break;
  case X86::PUNPCKHDQrr:
  case X86::VPUNPCKHDQrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::PUNPCKHDQrm:
  case X86::VPUNPCKHDQrm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKHMask(MVT::v4i32, ShuffleMask);
    break;
  case X86::VPUNPCKHDQYrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VPUNPCKHDQYrm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKHMask(MVT::v8i32, ShuffleMask);
    break;
  case X86::VPUNPCKHDQZrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VPUNPCKHDQZrm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKHMask(MVT::v16i32, ShuffleMask);
    break;
  case X86::PUNPCKHQDQrr:
  case X86::VPUNPCKHQDQrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::PUNPCKHQDQrm:
  case X86::VPUNPCKHQDQrm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKHMask(MVT::v2i64, ShuffleMask);
    break;
  case X86::VPUNPCKHQDQYrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VPUNPCKHQDQYrm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKHMask(MVT::v4i64, ShuffleMask);
    break;
  case X86::VPUNPCKHQDQZrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VPUNPCKHQDQZrm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKHMask(MVT::v8i64, ShuffleMask);
    break;

  case X86::MMX_PUNPCKLBWirr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
  case X86::MMX_PUNPCKLBWirm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKLMask(MVT::v8i8, ShuffleMask);
    break;
  case X86::MMX_PUNPCKLWDirr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
  case X86::MMX_PUNPCKLWDirm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKLMask(MVT::v4i16, ShuffleMask);
    break;
  case X86::MMX_PUNPCKLDQirr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
  case X86::MMX_PUNPCKLDQirm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKLMask(MVT::v2i32, ShuffleMask);
    break;
  case X86::PUNPCKLBWrr:
  case X86::VPUNPCKLBWrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::PUNPCKLBWrm:
  case X86::VPUNPCKLBWrm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKLMask(MVT::v16i8, ShuffleMask);
    break;
  case X86::VPUNPCKLBWYrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VPUNPCKLBWYrm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKLMask(MVT::v32i8, ShuffleMask);
    break;
  case X86::PUNPCKLWDrr:
  case X86::VPUNPCKLWDrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::PUNPCKLWDrm:
  case X86::VPUNPCKLWDrm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKLMask(MVT::v8i16, ShuffleMask);
    break;
  case X86::VPUNPCKLWDYrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VPUNPCKLWDYrm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKLMask(MVT::v16i16, ShuffleMask);
    break;
  case X86::PUNPCKLDQrr:
  case X86::VPUNPCKLDQrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::PUNPCKLDQrm:
  case X86::VPUNPCKLDQrm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKLMask(MVT::v4i32, ShuffleMask);
    break;
  case X86::VPUNPCKLDQYrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VPUNPCKLDQYrm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKLMask(MVT::v8i32, ShuffleMask);
    break;
  case X86::VPUNPCKLDQZrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VPUNPCKLDQZrm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKLMask(MVT::v16i32, ShuffleMask);
    break;
  case X86::PUNPCKLQDQrr:
  case X86::VPUNPCKLQDQrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::PUNPCKLQDQrm:
  case X86::VPUNPCKLQDQrm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKLMask(MVT::v2i64, ShuffleMask);
    break;
  case X86::VPUNPCKLQDQYrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VPUNPCKLQDQYrm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKLMask(MVT::v4i64, ShuffleMask);
    break;
  case X86::VPUNPCKLQDQZrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VPUNPCKLQDQZrm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKLMask(MVT::v8i64, ShuffleMask);
    break;

  case X86::SHUFPDrri:
  case X86::VSHUFPDrri:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::SHUFPDrmi:
  case X86::VSHUFPDrmi:
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodeSHUFPMask(MVT::v2f64,
                      MI->getOperand(MI->getNumOperands() - 1).getImm(),
                      ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VSHUFPDYrri:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VSHUFPDYrmi:
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodeSHUFPMask(MVT::v4f64,
                      MI->getOperand(MI->getNumOperands() - 1).getImm(),
                      ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  case X86::SHUFPSrri:
  case X86::VSHUFPSrri:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::SHUFPSrmi:
  case X86::VSHUFPSrmi:
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodeSHUFPMask(MVT::v4f32,
                      MI->getOperand(MI->getNumOperands() - 1).getImm(),
                      ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VSHUFPSYrri:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VSHUFPSYrmi:
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodeSHUFPMask(MVT::v8f32,
                      MI->getOperand(MI->getNumOperands() - 1).getImm(),
                      ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  CASE_VSHUF(64X2, r)
  CASE_VSHUF(64X2, m)
  CASE_VSHUF(32X4, r)
  CASE_VSHUF(32X4, m) {
    MVT VT;
    bool HasMemOp;
    unsigned NumOp = MI->getNumOperands();
    getVSHUF64x2FamilyInfo(MI, VT, HasMemOp);
    decodeVSHUF64x2FamilyMask(VT, MI->getOperand(NumOp - 1).getImm(),
                              ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    if (HasMemOp) {
      assert((NumOp >= 8) && "Expected at least 8 operands!");
      Src1Name = getRegName(MI->getOperand(NumOp - 7).getReg());
    } else {
      assert((NumOp >= 4) && "Expected at least 4 operands!");
      Src2Name = getRegName(MI->getOperand(NumOp - 2).getReg());
      Src1Name = getRegName(MI->getOperand(NumOp - 3).getReg());
    }
    break;
  }
  case X86::UNPCKLPDrr:
  case X86::VUNPCKLPDrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::UNPCKLPDrm:
  case X86::VUNPCKLPDrm:
    DecodeUNPCKLMask(MVT::v2f64, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VUNPCKLPDYrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VUNPCKLPDYrm:
    DecodeUNPCKLMask(MVT::v4f64, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VUNPCKLPDZrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VUNPCKLPDZrm:
    DecodeUNPCKLMask(MVT::v8f64, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::UNPCKLPSrr:
  case X86::VUNPCKLPSrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::UNPCKLPSrm:
  case X86::VUNPCKLPSrm:
    DecodeUNPCKLMask(MVT::v4f32, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VUNPCKLPSYrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VUNPCKLPSYrm:
    DecodeUNPCKLMask(MVT::v8f32, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VUNPCKLPSZrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VUNPCKLPSZrm:
    DecodeUNPCKLMask(MVT::v16f32, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::UNPCKHPDrr:
  case X86::VUNPCKHPDrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::UNPCKHPDrm:
  case X86::VUNPCKHPDrm:
    DecodeUNPCKHMask(MVT::v2f64, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VUNPCKHPDYrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VUNPCKHPDYrm:
    DecodeUNPCKHMask(MVT::v4f64, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VUNPCKHPDZrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VUNPCKHPDZrm:
    DecodeUNPCKHMask(MVT::v8f64, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::UNPCKHPSrr:
  case X86::VUNPCKHPSrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::UNPCKHPSrm:
  case X86::VUNPCKHPSrm:
    DecodeUNPCKHMask(MVT::v4f32, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VUNPCKHPSYrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VUNPCKHPSYrm:
    DecodeUNPCKHMask(MVT::v8f32, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VUNPCKHPSZrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VUNPCKHPSZrm:
    DecodeUNPCKHMask(MVT::v16f32, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VPERMILPSri:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    // FALL THROUGH.
  case X86::VPERMILPSmi:
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodePSHUFMask(MVT::v4f32,
                      MI->getOperand(MI->getNumOperands() - 1).getImm(),
                      ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VPERMILPSYri:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    // FALL THROUGH.
  case X86::VPERMILPSYmi:
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodePSHUFMask(MVT::v8f32,
                      MI->getOperand(MI->getNumOperands() - 1).getImm(),
                      ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VPERMILPDri:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    // FALL THROUGH.
  case X86::VPERMILPDmi:
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodePSHUFMask(MVT::v2f64,
                      MI->getOperand(MI->getNumOperands() - 1).getImm(),
                      ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VPERMILPDYri:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    // FALL THROUGH.
  case X86::VPERMILPDYmi:
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodePSHUFMask(MVT::v4f64,
                      MI->getOperand(MI->getNumOperands() - 1).getImm(),
                      ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VPERM2F128rr:
  case X86::VPERM2I128rr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VPERM2F128rm:
  case X86::VPERM2I128rm:
    // For instruction comments purpose, assume the 256-bit vector is v4i64.
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodeVPERM2X128Mask(MVT::v4i64,
                           MI->getOperand(MI->getNumOperands() - 1).getImm(),
                           ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VPERMQYri:
  case X86::VPERMPDYri:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    // FALL THROUGH.
  case X86::VPERMQYmi:
  case X86::VPERMPDYmi:
    if (MI->getOperand(MI->getNumOperands() - 1).isImm())
      DecodeVPERMMask(MI->getOperand(MI->getNumOperands() - 1).getImm(),
                      ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  case X86::MOVSDrr:
  case X86::VMOVSDrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    Src1Name = getRegName(MI->getOperand(1).getReg());
    // FALL THROUGH.
  case X86::MOVSDrm:
  case X86::VMOVSDrm:
    DecodeScalarMoveMask(MVT::v2f64, nullptr == Src2Name, ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::MOVSSrr:
  case X86::VMOVSSrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    Src1Name = getRegName(MI->getOperand(1).getReg());
    // FALL THROUGH.
  case X86::MOVSSrm:
  case X86::VMOVSSrm:
    DecodeScalarMoveMask(MVT::v4f32, nullptr == Src2Name, ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  case X86::MOVPQI2QIrr:
  case X86::MOVZPQILo2PQIrr:
  case X86::VMOVPQI2QIrr:
  case X86::VMOVZPQILo2PQIrr:
    Src1Name = getRegName(MI->getOperand(1).getReg());
  // FALL THROUGH.
  case X86::MOVQI2PQIrm:
  case X86::MOVZQI2PQIrm:
  case X86::MOVZPQILo2PQIrm:
  case X86::VMOVQI2PQIrm:
  case X86::VMOVZQI2PQIrm:
  case X86::VMOVZPQILo2PQIrm:
    DecodeZeroMoveLowMask(MVT::v2i64, ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::MOVDI2PDIrm:
  case X86::VMOVDI2PDIrm:
    DecodeZeroMoveLowMask(MVT::v4i32, ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  case X86::EXTRQI:
    if (MI->getOperand(2).isImm() &&
        MI->getOperand(3).isImm())
      DecodeEXTRQIMask(MI->getOperand(2).getImm(),
                       MI->getOperand(3).getImm(),
                       ShuffleMask);

    DestName = getRegName(MI->getOperand(0).getReg());
    Src1Name = getRegName(MI->getOperand(1).getReg());
    break;

  case X86::INSERTQI:
    if (MI->getOperand(3).isImm() &&
        MI->getOperand(4).isImm())
      DecodeINSERTQIMask(MI->getOperand(3).getImm(),
                         MI->getOperand(4).getImm(),
                         ShuffleMask);

    DestName = getRegName(MI->getOperand(0).getReg());
    Src1Name = getRegName(MI->getOperand(1).getReg());
    Src2Name = getRegName(MI->getOperand(2).getReg());
    break;

  case X86::PMOVZXBWrr:
  case X86::PMOVZXBDrr:
  case X86::PMOVZXBQrr:
  case X86::PMOVZXWDrr:
  case X86::PMOVZXWQrr:
  case X86::PMOVZXDQrr:
  case X86::VPMOVZXBWrr:
  case X86::VPMOVZXBDrr:
  case X86::VPMOVZXBQrr:
  case X86::VPMOVZXWDrr:
  case X86::VPMOVZXWQrr:
  case X86::VPMOVZXDQrr:
  case X86::VPMOVZXBWYrr:
  case X86::VPMOVZXBDYrr:
  case X86::VPMOVZXBQYrr:
  case X86::VPMOVZXWDYrr:
  case X86::VPMOVZXWQYrr:
  case X86::VPMOVZXDQYrr:
    Src1Name = getRegName(MI->getOperand(1).getReg());
  // FALL THROUGH.
  case X86::PMOVZXBWrm:
  case X86::PMOVZXBDrm:
  case X86::PMOVZXBQrm:
  case X86::PMOVZXWDrm:
  case X86::PMOVZXWQrm:
  case X86::PMOVZXDQrm:
  case X86::VPMOVZXBWrm:
  case X86::VPMOVZXBDrm:
  case X86::VPMOVZXBQrm:
  case X86::VPMOVZXWDrm:
  case X86::VPMOVZXWQrm:
  case X86::VPMOVZXDQrm:
  case X86::VPMOVZXBWYrm:
  case X86::VPMOVZXBDYrm:
  case X86::VPMOVZXBQYrm:
  case X86::VPMOVZXWDYrm:
  case X86::VPMOVZXWQYrm:
  case X86::VPMOVZXDQYrm: {
    MVT SrcVT, DstVT;
    getZeroExtensionTypes(MI, SrcVT, DstVT);
    DecodeZeroExtendMask(SrcVT, DstVT, ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
  } break;
  }

  // The only comments we decode are shuffles, so give up if we were unable to
  // decode a shuffle mask.
  if (ShuffleMask.empty())
    return false;

  if (!DestName) DestName = Src1Name;
  OS << (DestName ? DestName : "mem") << " = ";

  // If the two sources are the same, canonicalize the input elements to be
  // from the first src so that we get larger element spans.
  if (Src1Name == Src2Name) {
    for (unsigned i = 0, e = ShuffleMask.size(); i != e; ++i) {
      if ((int)ShuffleMask[i] >= 0 && // Not sentinel.
          ShuffleMask[i] >= (int)e)   // From second mask.
        ShuffleMask[i] -= e;
    }
  }

  // The shuffle mask specifies which elements of the src1/src2 fill in the
  // destination, with a few sentinel values.  Loop through and print them
  // out.
  for (unsigned i = 0, e = ShuffleMask.size(); i != e; ++i) {
    if (i != 0)
      OS << ',';
    if (ShuffleMask[i] == SM_SentinelZero) {
      OS << "zero";
      continue;
    }

    // Otherwise, it must come from src1 or src2.  Print the span of elements
    // that comes from this src.
    bool isSrc1 = ShuffleMask[i] < (int)ShuffleMask.size();
    const char *SrcName = isSrc1 ? Src1Name : Src2Name;
    OS << (SrcName ? SrcName : "mem") << '[';
    bool IsFirst = true;
    while (i != e && (int)ShuffleMask[i] != SM_SentinelZero &&
           (ShuffleMask[i] < (int)ShuffleMask.size()) == isSrc1) {
      if (!IsFirst)
        OS << ',';
      else
        IsFirst = false;
      if (ShuffleMask[i] == SM_SentinelUndef)
        OS << "u";
      else
        OS << ShuffleMask[i] % ShuffleMask.size();
      ++i;
    }
    OS << ']';
    --i; // For loop increments element #.
  }
  //MI->print(OS, 0);
  OS << "\n";

  // We successfully added a comment to this instruction.
  return true;
}
