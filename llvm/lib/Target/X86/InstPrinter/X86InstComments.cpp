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
#include "llvm/CodeGen/MachineValueType.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define CASE_SSE_INS_COMMON(Inst, src)            \
  case X86::Inst##src:

#define CASE_AVX_INS_COMMON(Inst, Suffix, src)    \
  case X86::V##Inst##Suffix##src:

#define CASE_MASK_INS_COMMON(Inst, Suffix, src)   \
  case X86::V##Inst##Suffix##src##k:

#define CASE_MASKZ_INS_COMMON(Inst, Suffix, src)  \
  case X86::V##Inst##Suffix##src##kz:

#define CASE_AVX512_INS_COMMON(Inst, Suffix, src) \
  CASE_AVX_INS_COMMON(Inst, Suffix, src)          \
  CASE_MASK_INS_COMMON(Inst, Suffix, src)         \
  CASE_MASKZ_INS_COMMON(Inst, Suffix, src)

#define CASE_MOVDUP(Inst, src)                    \
  CASE_AVX512_INS_COMMON(Inst, Z, r##src)         \
  CASE_AVX512_INS_COMMON(Inst, Z256, r##src)      \
  CASE_AVX512_INS_COMMON(Inst, Z128, r##src)      \
  CASE_AVX_INS_COMMON(Inst, , r##src)             \
  CASE_AVX_INS_COMMON(Inst, Y, r##src)            \
  CASE_SSE_INS_COMMON(Inst, r##src)

#define CASE_MASK_MOVDUP(Inst, src)               \
  CASE_MASK_INS_COMMON(Inst, Z, r##src)           \
  CASE_MASK_INS_COMMON(Inst, Z256, r##src)        \
  CASE_MASK_INS_COMMON(Inst, Z128, r##src)

#define CASE_MASKZ_MOVDUP(Inst, src)              \
  CASE_MASKZ_INS_COMMON(Inst, Z, r##src)          \
  CASE_MASKZ_INS_COMMON(Inst, Z256, r##src)       \
  CASE_MASKZ_INS_COMMON(Inst, Z128, r##src)

#define CASE_PMOVZX(Inst, src)                    \
  CASE_AVX512_INS_COMMON(Inst, Z, r##src)         \
  CASE_AVX512_INS_COMMON(Inst, Z256, r##src)      \
  CASE_AVX512_INS_COMMON(Inst, Z128, r##src)      \
  CASE_AVX_INS_COMMON(Inst, , r##src)             \
  CASE_AVX_INS_COMMON(Inst, Y, r##src)            \
  CASE_SSE_INS_COMMON(Inst, r##src)

#define CASE_MASK_PMOVZX(Inst, src)               \
  CASE_MASK_INS_COMMON(Inst, Z, r##src)           \
  CASE_MASK_INS_COMMON(Inst, Z256, r##src)        \
  CASE_MASK_INS_COMMON(Inst, Z128, r##src)

#define CASE_MASKZ_PMOVZX(Inst, src)              \
  CASE_MASKZ_INS_COMMON(Inst, Z, r##src)          \
  CASE_MASKZ_INS_COMMON(Inst, Z256, r##src)       \
  CASE_MASKZ_INS_COMMON(Inst, Z128, r##src)

#define CASE_UNPCK(Inst, src)                     \
  CASE_AVX512_INS_COMMON(Inst, Z, r##src)         \
  CASE_AVX512_INS_COMMON(Inst, Z256, r##src)      \
  CASE_AVX512_INS_COMMON(Inst, Z128, r##src)      \
  CASE_AVX_INS_COMMON(Inst, , r##src)             \
  CASE_AVX_INS_COMMON(Inst, Y, r##src)            \
  CASE_SSE_INS_COMMON(Inst, r##src)

#define CASE_MASK_UNPCK(Inst, src)                \
  CASE_MASK_INS_COMMON(Inst, Z, r##src)           \
  CASE_MASK_INS_COMMON(Inst, Z256, r##src)        \
  CASE_MASK_INS_COMMON(Inst, Z128, r##src)

#define CASE_MASKZ_UNPCK(Inst, src)               \
  CASE_MASKZ_INS_COMMON(Inst, Z, r##src)          \
  CASE_MASKZ_INS_COMMON(Inst, Z256, r##src)       \
  CASE_MASKZ_INS_COMMON(Inst, Z128, r##src)

#define CASE_SHUF(Inst, suf)                      \
  CASE_AVX512_INS_COMMON(Inst, Z, suf)            \
  CASE_AVX512_INS_COMMON(Inst, Z256, suf)         \
  CASE_AVX512_INS_COMMON(Inst, Z128, suf)         \
  CASE_AVX_INS_COMMON(Inst, , suf)                \
  CASE_AVX_INS_COMMON(Inst, Y, suf)               \
  CASE_SSE_INS_COMMON(Inst, suf)

#define CASE_MASK_SHUF(Inst, src)                 \
  CASE_MASK_INS_COMMON(Inst, Z, r##src##i)        \
  CASE_MASK_INS_COMMON(Inst, Z256, r##src##i)     \
  CASE_MASK_INS_COMMON(Inst, Z128, r##src##i)

#define CASE_MASKZ_SHUF(Inst, src)                \
  CASE_MASKZ_INS_COMMON(Inst, Z, r##src##i)       \
  CASE_MASKZ_INS_COMMON(Inst, Z256, r##src##i)    \
  CASE_MASKZ_INS_COMMON(Inst, Z128, r##src##i)

#define CASE_VPERMILPI(Inst, src)                 \
  CASE_AVX512_INS_COMMON(Inst, Z, src##i)         \
  CASE_AVX512_INS_COMMON(Inst, Z256, src##i)      \
  CASE_AVX512_INS_COMMON(Inst, Z128, src##i)      \
  CASE_AVX_INS_COMMON(Inst, , src##i)             \
  CASE_AVX_INS_COMMON(Inst, Y, src##i)

#define CASE_MASK_VPERMILPI(Inst, src)            \
  CASE_MASK_INS_COMMON(Inst, Z, src##i)           \
  CASE_MASK_INS_COMMON(Inst, Z256, src##i)        \
  CASE_MASK_INS_COMMON(Inst, Z128, src##i)

#define CASE_MASKZ_VPERMILPI(Inst, src)           \
  CASE_MASKZ_INS_COMMON(Inst, Z, src##i)          \
  CASE_MASKZ_INS_COMMON(Inst, Z256, src##i)       \
  CASE_MASKZ_INS_COMMON(Inst, Z128, src##i)

#define CASE_VPERM(Inst, src)                     \
  CASE_AVX512_INS_COMMON(Inst, Z, src##i)         \
  CASE_AVX512_INS_COMMON(Inst, Z256, src##i)      \
  CASE_AVX_INS_COMMON(Inst, Y, src##i)

#define CASE_MASK_VPERM(Inst, src)                \
  CASE_MASK_INS_COMMON(Inst, Z, src##i)           \
  CASE_MASK_INS_COMMON(Inst, Z256, src##i)

#define CASE_MASKZ_VPERM(Inst, src)               \
  CASE_MASKZ_INS_COMMON(Inst, Z, src##i)          \
  CASE_MASKZ_INS_COMMON(Inst, Z256, src##i)

#define CASE_VSHUF(Inst, src)                          \
  CASE_AVX512_INS_COMMON(SHUFF##Inst, Z, r##src##i)    \
  CASE_AVX512_INS_COMMON(SHUFI##Inst, Z, r##src##i)    \
  CASE_AVX512_INS_COMMON(SHUFF##Inst, Z256, r##src##i) \
  CASE_AVX512_INS_COMMON(SHUFI##Inst, Z256, r##src##i)

#define CASE_MASK_VSHUF(Inst, src)                    \
  CASE_MASK_INS_COMMON(SHUFF##Inst, Z, r##src##i)     \
  CASE_MASK_INS_COMMON(SHUFI##Inst, Z, r##src##i)     \
  CASE_MASK_INS_COMMON(SHUFF##Inst, Z256, r##src##i)  \
  CASE_MASK_INS_COMMON(SHUFI##Inst, Z256, r##src##i)

#define CASE_MASKZ_VSHUF(Inst, src)                   \
  CASE_MASKZ_INS_COMMON(SHUFF##Inst, Z, r##src##i)    \
  CASE_MASKZ_INS_COMMON(SHUFI##Inst, Z, r##src##i)    \
  CASE_MASKZ_INS_COMMON(SHUFF##Inst, Z256, r##src##i) \
  CASE_MASKZ_INS_COMMON(SHUFI##Inst, Z256, r##src##i)

static unsigned getVectorRegSize(unsigned RegNo) {
  if (X86::ZMM0 <= RegNo && RegNo <= X86::ZMM31)
    return 512;
  if (X86::YMM0 <= RegNo && RegNo <= X86::YMM31)
    return 256;
  if (X86::XMM0 <= RegNo && RegNo <= X86::XMM31)
    return 128;
  if (X86::MM0 <= RegNo && RegNo <= X86::MM7)
    return 64;

  llvm_unreachable("Unknown vector reg!");
}

static MVT getRegOperandVectorVT(const MCInst *MI, const MVT &ScalarVT,
                                 unsigned OperandIndex) {
  unsigned OpReg = MI->getOperand(OperandIndex).getReg();
  return MVT::getVectorVT(ScalarVT,
                          getVectorRegSize(OpReg)/ScalarVT.getSizeInBits());
}

/// \brief Extracts the dst type for a given zero extension instruction.
static MVT getZeroExtensionResultType(const MCInst *MI) {
  switch (MI->getOpcode()) {
  default:
    llvm_unreachable("Unknown zero extension instruction");
  // zero extension to i16
  CASE_PMOVZX(PMOVZXBW, m)
  CASE_PMOVZX(PMOVZXBW, r)
    return getRegOperandVectorVT(MI, MVT::i16, 0);
  // zero extension to i32
  CASE_PMOVZX(PMOVZXBD, m)
  CASE_PMOVZX(PMOVZXBD, r)
  CASE_PMOVZX(PMOVZXWD, m)
  CASE_PMOVZX(PMOVZXWD, r)
    return getRegOperandVectorVT(MI, MVT::i32, 0);
  // zero extension to i64
  CASE_PMOVZX(PMOVZXBQ, m)
  CASE_PMOVZX(PMOVZXBQ, r)
  CASE_PMOVZX(PMOVZXWQ, m)
  CASE_PMOVZX(PMOVZXWQ, r)
  CASE_PMOVZX(PMOVZXDQ, m)
  CASE_PMOVZX(PMOVZXDQ, r)
    return getRegOperandVectorVT(MI, MVT::i64, 0);
  }
}

/// Wraps the destination register name with AVX512 mask/maskz filtering.
static std::string getMaskName(const MCInst *MI, const char *DestName,
                               const char *(*getRegName)(unsigned)) {
  std::string OpMaskName(DestName);

  bool MaskWithZero = false;
  const char *MaskRegName = nullptr;

  switch (MI->getOpcode()) {
  default:
    return OpMaskName;
  CASE_MASKZ_MOVDUP(MOVDDUP, m)
  CASE_MASKZ_MOVDUP(MOVDDUP, r)
  CASE_MASKZ_MOVDUP(MOVSHDUP, m)
  CASE_MASKZ_MOVDUP(MOVSHDUP, r)
  CASE_MASKZ_MOVDUP(MOVSLDUP, m)
  CASE_MASKZ_MOVDUP(MOVSLDUP, r)
  CASE_MASKZ_PMOVZX(PMOVZXBD, m)
  CASE_MASKZ_PMOVZX(PMOVZXBD, r)
  CASE_MASKZ_PMOVZX(PMOVZXBQ, m)
  CASE_MASKZ_PMOVZX(PMOVZXBQ, r)
  CASE_MASKZ_PMOVZX(PMOVZXBW, m)
  CASE_MASKZ_PMOVZX(PMOVZXBW, r)
  CASE_MASKZ_PMOVZX(PMOVZXDQ, m)
  CASE_MASKZ_PMOVZX(PMOVZXDQ, r)
  CASE_MASKZ_PMOVZX(PMOVZXWD, m)
  CASE_MASKZ_PMOVZX(PMOVZXWD, r)
  CASE_MASKZ_PMOVZX(PMOVZXWQ, m)
  CASE_MASKZ_PMOVZX(PMOVZXWQ, r)
  CASE_MASKZ_UNPCK(PUNPCKHBW, m)
  CASE_MASKZ_UNPCK(PUNPCKHBW, r)
  CASE_MASKZ_UNPCK(PUNPCKHWD, m)
  CASE_MASKZ_UNPCK(PUNPCKHWD, r)
  CASE_MASKZ_UNPCK(PUNPCKHDQ, m)
  CASE_MASKZ_UNPCK(PUNPCKHDQ, r)
  CASE_MASKZ_UNPCK(PUNPCKLBW, m)
  CASE_MASKZ_UNPCK(PUNPCKLBW, r)
  CASE_MASKZ_UNPCK(PUNPCKLWD, m)
  CASE_MASKZ_UNPCK(PUNPCKLWD, r)
  CASE_MASKZ_UNPCK(PUNPCKLDQ, m)
  CASE_MASKZ_UNPCK(PUNPCKLDQ, r)
  CASE_MASKZ_UNPCK(UNPCKHPD, m)
  CASE_MASKZ_UNPCK(UNPCKHPD, r)
  CASE_MASKZ_UNPCK(UNPCKHPS, m)
  CASE_MASKZ_UNPCK(UNPCKHPS, r)
  CASE_MASKZ_UNPCK(UNPCKLPD, m)
  CASE_MASKZ_UNPCK(UNPCKLPD, r)
  CASE_MASKZ_UNPCK(UNPCKLPS, m)
  CASE_MASKZ_UNPCK(UNPCKLPS, r)
  CASE_MASKZ_SHUF(PALIGNR, r)
  CASE_MASKZ_SHUF(PALIGNR, m)
  CASE_MASKZ_SHUF(ALIGNQ, r)
  CASE_MASKZ_SHUF(ALIGNQ, m)
  CASE_MASKZ_SHUF(ALIGND, r)
  CASE_MASKZ_SHUF(ALIGND, m)
  CASE_MASKZ_SHUF(SHUFPD, m)
  CASE_MASKZ_SHUF(SHUFPD, r)
  CASE_MASKZ_SHUF(SHUFPS, m)
  CASE_MASKZ_SHUF(SHUFPS, r)
  CASE_MASKZ_VPERMILPI(PERMILPD, m)
  CASE_MASKZ_VPERMILPI(PERMILPD, r)
  CASE_MASKZ_VPERMILPI(PERMILPS, m)
  CASE_MASKZ_VPERMILPI(PERMILPS, r)
  CASE_MASKZ_VPERMILPI(PSHUFD, m)
  CASE_MASKZ_VPERMILPI(PSHUFD, r)
  CASE_MASKZ_VPERMILPI(PSHUFHW, m)
  CASE_MASKZ_VPERMILPI(PSHUFHW, r)
  CASE_MASKZ_VPERMILPI(PSHUFLW, m)
  CASE_MASKZ_VPERMILPI(PSHUFLW, r)
  CASE_MASKZ_VPERM(PERMPD, m)
  CASE_MASKZ_VPERM(PERMPD, r)
  CASE_MASKZ_VPERM(PERMQ, m)
  CASE_MASKZ_VPERM(PERMQ, r)
  CASE_MASKZ_VSHUF(64X2, m)
  CASE_MASKZ_VSHUF(64X2, r)
  CASE_MASKZ_VSHUF(32X4, m)
  CASE_MASKZ_VSHUF(32X4, r)
  CASE_MASKZ_INS_COMMON(BROADCASTF64X2, Z128, rm)
  CASE_MASKZ_INS_COMMON(BROADCASTI64X2, Z128, rm)
  CASE_MASKZ_INS_COMMON(BROADCASTF64X2, , rm)
  CASE_MASKZ_INS_COMMON(BROADCASTI64X2, , rm)
  CASE_MASKZ_INS_COMMON(BROADCASTF64X4, , rm)
  CASE_MASKZ_INS_COMMON(BROADCASTI64X4, , rm)
  CASE_MASKZ_INS_COMMON(BROADCASTF32X4, Z256, rm)
  CASE_MASKZ_INS_COMMON(BROADCASTI32X4, Z256, rm)
  CASE_MASKZ_INS_COMMON(BROADCASTF32X4, , rm)
  CASE_MASKZ_INS_COMMON(BROADCASTI32X4, , rm)
  CASE_MASKZ_INS_COMMON(BROADCASTF32X8, , rm)
  CASE_MASKZ_INS_COMMON(BROADCASTI32X8, , rm)
  CASE_MASKZ_INS_COMMON(BROADCASTF32X2, Z256, r)
  CASE_MASKZ_INS_COMMON(BROADCASTI32X2, Z256, r)
  CASE_MASKZ_INS_COMMON(BROADCASTF32X2, Z256, m)
  CASE_MASKZ_INS_COMMON(BROADCASTI32X2, Z256, m)
  CASE_MASKZ_INS_COMMON(BROADCASTF32X2, Z, r)
  CASE_MASKZ_INS_COMMON(BROADCASTI32X2, Z, r)
  CASE_MASKZ_INS_COMMON(BROADCASTF32X2, Z, m)
  CASE_MASKZ_INS_COMMON(BROADCASTI32X2, Z, m)
    MaskWithZero = true;
    MaskRegName = getRegName(MI->getOperand(1).getReg());
    break;
  CASE_MASK_MOVDUP(MOVDDUP, m)
  CASE_MASK_MOVDUP(MOVDDUP, r)
  CASE_MASK_MOVDUP(MOVSHDUP, m)
  CASE_MASK_MOVDUP(MOVSHDUP, r)
  CASE_MASK_MOVDUP(MOVSLDUP, m)
  CASE_MASK_MOVDUP(MOVSLDUP, r)
  CASE_MASK_PMOVZX(PMOVZXBD, m)
  CASE_MASK_PMOVZX(PMOVZXBD, r)
  CASE_MASK_PMOVZX(PMOVZXBQ, m)
  CASE_MASK_PMOVZX(PMOVZXBQ, r)
  CASE_MASK_PMOVZX(PMOVZXBW, m)
  CASE_MASK_PMOVZX(PMOVZXBW, r)
  CASE_MASK_PMOVZX(PMOVZXDQ, m)
  CASE_MASK_PMOVZX(PMOVZXDQ, r)
  CASE_MASK_PMOVZX(PMOVZXWD, m)
  CASE_MASK_PMOVZX(PMOVZXWD, r)
  CASE_MASK_PMOVZX(PMOVZXWQ, m)
  CASE_MASK_PMOVZX(PMOVZXWQ, r)
  CASE_MASK_UNPCK(PUNPCKHBW, m)
  CASE_MASK_UNPCK(PUNPCKHBW, r)
  CASE_MASK_UNPCK(PUNPCKHWD, m)
  CASE_MASK_UNPCK(PUNPCKHWD, r)
  CASE_MASK_UNPCK(PUNPCKHDQ, m)
  CASE_MASK_UNPCK(PUNPCKHDQ, r)
  CASE_MASK_UNPCK(PUNPCKLBW, m)
  CASE_MASK_UNPCK(PUNPCKLBW, r)
  CASE_MASK_UNPCK(PUNPCKLWD, m)
  CASE_MASK_UNPCK(PUNPCKLWD, r)
  CASE_MASK_UNPCK(PUNPCKLDQ, m)
  CASE_MASK_UNPCK(PUNPCKLDQ, r)
  CASE_MASK_UNPCK(UNPCKHPD, m)
  CASE_MASK_UNPCK(UNPCKHPD, r)
  CASE_MASK_UNPCK(UNPCKHPS, m)
  CASE_MASK_UNPCK(UNPCKHPS, r)
  CASE_MASK_UNPCK(UNPCKLPD, m)
  CASE_MASK_UNPCK(UNPCKLPD, r)
  CASE_MASK_UNPCK(UNPCKLPS, m)
  CASE_MASK_UNPCK(UNPCKLPS, r)
  CASE_MASK_SHUF(PALIGNR, r)
  CASE_MASK_SHUF(PALIGNR, m)
  CASE_MASK_SHUF(ALIGNQ, r)
  CASE_MASK_SHUF(ALIGNQ, m)
  CASE_MASK_SHUF(ALIGND, r)
  CASE_MASK_SHUF(ALIGND, m)
  CASE_MASK_SHUF(SHUFPD, m)
  CASE_MASK_SHUF(SHUFPD, r)
  CASE_MASK_SHUF(SHUFPS, m)
  CASE_MASK_SHUF(SHUFPS, r)
  CASE_MASK_VPERMILPI(PERMILPD, m)
  CASE_MASK_VPERMILPI(PERMILPD, r)
  CASE_MASK_VPERMILPI(PERMILPS, m)
  CASE_MASK_VPERMILPI(PERMILPS, r)
  CASE_MASK_VPERMILPI(PSHUFD, m)
  CASE_MASK_VPERMILPI(PSHUFD, r)
  CASE_MASK_VPERMILPI(PSHUFHW, m)
  CASE_MASK_VPERMILPI(PSHUFHW, r)
  CASE_MASK_VPERMILPI(PSHUFLW, m)
  CASE_MASK_VPERMILPI(PSHUFLW, r)
  CASE_MASK_VPERM(PERMPD, m)
  CASE_MASK_VPERM(PERMPD, r)
  CASE_MASK_VPERM(PERMQ, m)
  CASE_MASK_VPERM(PERMQ, r)
  CASE_MASK_VSHUF(64X2, m)
  CASE_MASK_VSHUF(64X2, r)
  CASE_MASK_VSHUF(32X4, m)
  CASE_MASK_VSHUF(32X4, r)
  CASE_MASK_INS_COMMON(BROADCASTF64X2, Z128, rm)
  CASE_MASK_INS_COMMON(BROADCASTI64X2, Z128, rm)
  CASE_MASK_INS_COMMON(BROADCASTF64X2, , rm)
  CASE_MASK_INS_COMMON(BROADCASTI64X2, , rm)
  CASE_MASK_INS_COMMON(BROADCASTF64X4, , rm)
  CASE_MASK_INS_COMMON(BROADCASTI64X4, , rm)
  CASE_MASK_INS_COMMON(BROADCASTF32X4, Z256, rm)
  CASE_MASK_INS_COMMON(BROADCASTI32X4, Z256, rm)
  CASE_MASK_INS_COMMON(BROADCASTF32X4, , rm)
  CASE_MASK_INS_COMMON(BROADCASTI32X4, , rm)
  CASE_MASK_INS_COMMON(BROADCASTF32X8, , rm)
  CASE_MASK_INS_COMMON(BROADCASTI32X8, , rm)
  CASE_MASK_INS_COMMON(BROADCASTF32X2, Z256, r)
  CASE_MASK_INS_COMMON(BROADCASTI32X2, Z256, r)
  CASE_MASK_INS_COMMON(BROADCASTF32X2, Z256, m)
  CASE_MASK_INS_COMMON(BROADCASTI32X2, Z256, m)
  CASE_MASK_INS_COMMON(BROADCASTF32X2, Z, r)
  CASE_MASK_INS_COMMON(BROADCASTI32X2, Z, r)
  CASE_MASK_INS_COMMON(BROADCASTF32X2, Z, m)
  CASE_MASK_INS_COMMON(BROADCASTI32X2, Z, m)
    MaskRegName = getRegName(MI->getOperand(2).getReg());
    break;
  }

  // MASK: zmmX {%kY}
  OpMaskName += " {%";
  OpMaskName += MaskRegName;
  OpMaskName += "}";

  // MASKZ: zmmX {%kY} {z}
  if (MaskWithZero)
    OpMaskName += " {z}";

  return OpMaskName;
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
  unsigned NumOperands = MI->getNumOperands();
  bool RegForm = false;

  switch (MI->getOpcode()) {
  default:
    // Not an instruction for which we can decode comments.
    return false;

  case X86::BLENDPDrri:
  case X86::VBLENDPDrri:
  case X86::VBLENDPDYrri:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    LLVM_FALLTHROUGH;
  case X86::BLENDPDrmi:
  case X86::VBLENDPDrmi:
  case X86::VBLENDPDYrmi:
    if (MI->getOperand(NumOperands - 1).isImm())
      DecodeBLENDMask(getRegOperandVectorVT(MI, MVT::f64, 0),
                      MI->getOperand(NumOperands - 1).getImm(),
                      ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  case X86::BLENDPSrri:
  case X86::VBLENDPSrri:
  case X86::VBLENDPSYrri:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    LLVM_FALLTHROUGH;
  case X86::BLENDPSrmi:
  case X86::VBLENDPSrmi:
  case X86::VBLENDPSYrmi:
    if (MI->getOperand(NumOperands - 1).isImm())
      DecodeBLENDMask(getRegOperandVectorVT(MI, MVT::f32, 0),
                      MI->getOperand(NumOperands - 1).getImm(),
                      ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  case X86::PBLENDWrri:
  case X86::VPBLENDWrri:
  case X86::VPBLENDWYrri:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    LLVM_FALLTHROUGH;
  case X86::PBLENDWrmi:
  case X86::VPBLENDWrmi:
  case X86::VPBLENDWYrmi:
    if (MI->getOperand(NumOperands - 1).isImm())
      DecodeBLENDMask(getRegOperandVectorVT(MI, MVT::i16, 0),
                      MI->getOperand(NumOperands - 1).getImm(),
                      ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  case X86::VPBLENDDrri:
  case X86::VPBLENDDYrri:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    LLVM_FALLTHROUGH;
  case X86::VPBLENDDrmi:
  case X86::VPBLENDDYrmi:
    if (MI->getOperand(NumOperands - 1).isImm())
      DecodeBLENDMask(getRegOperandVectorVT(MI, MVT::i32, 0),
                      MI->getOperand(NumOperands - 1).getImm(),
                      ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  case X86::INSERTPSrr:
  case X86::VINSERTPSrr:
  case X86::VINSERTPSZrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    LLVM_FALLTHROUGH;
  case X86::INSERTPSrm:
  case X86::VINSERTPSrm:
  case X86::VINSERTPSZrm:
    DestName = getRegName(MI->getOperand(0).getReg());
    Src1Name = getRegName(MI->getOperand(1).getReg());
    if (MI->getOperand(NumOperands - 1).isImm())
      DecodeINSERTPSMask(MI->getOperand(NumOperands - 1).getImm(),
                         ShuffleMask);
    break;

  case X86::MOVLHPSrr:
  case X86::VMOVLHPSrr:
  case X86::VMOVLHPSZrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeMOVLHPSMask(2, ShuffleMask);
    break;

  case X86::MOVHLPSrr:
  case X86::VMOVHLPSrr:
  case X86::VMOVHLPSZrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeMOVHLPSMask(2, ShuffleMask);
    break;

  case X86::MOVHPDrm:
  case X86::VMOVHPDrm:
  case X86::VMOVHPDZ128rm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeInsertElementMask(MVT::v2f64, 1, 1, ShuffleMask);
    break;

  case X86::MOVHPSrm:
  case X86::VMOVHPSrm:
  case X86::VMOVHPSZ128rm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeInsertElementMask(MVT::v4f32, 2, 2, ShuffleMask);
    break;

  case X86::MOVLPDrm:
  case X86::VMOVLPDrm:
  case X86::VMOVLPDZ128rm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeInsertElementMask(MVT::v2f64, 0, 1, ShuffleMask);
    break;

  case X86::MOVLPSrm:
  case X86::VMOVLPSrm:
  case X86::VMOVLPSZ128rm:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeInsertElementMask(MVT::v4f32, 0, 2, ShuffleMask);
    break;

  CASE_MOVDUP(MOVSLDUP, r)
    Src1Name = getRegName(MI->getOperand(NumOperands - 1).getReg());
    LLVM_FALLTHROUGH;

  CASE_MOVDUP(MOVSLDUP, m)
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeMOVSLDUPMask(getRegOperandVectorVT(MI, MVT::f32, 0), ShuffleMask);
    break;

  CASE_MOVDUP(MOVSHDUP, r)
    Src1Name = getRegName(MI->getOperand(NumOperands - 1).getReg());
    LLVM_FALLTHROUGH;

  CASE_MOVDUP(MOVSHDUP, m)
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeMOVSHDUPMask(getRegOperandVectorVT(MI, MVT::f32, 0), ShuffleMask);
    break;

  CASE_MOVDUP(MOVDDUP, r)
    Src1Name = getRegName(MI->getOperand(NumOperands - 1).getReg());
    LLVM_FALLTHROUGH;

  CASE_MOVDUP(MOVDDUP, m)
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeMOVDDUPMask(getRegOperandVectorVT(MI, MVT::f64, 0), ShuffleMask);
    break;

  case X86::PSLLDQri:
  case X86::VPSLLDQri:
  case X86::VPSLLDQYri:
  case X86::VPSLLDQZ128rr:
  case X86::VPSLLDQZ256rr:
  case X86::VPSLLDQZ512rr:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    LLVM_FALLTHROUGH;
  case X86::VPSLLDQZ128rm:
  case X86::VPSLLDQZ256rm:
  case X86::VPSLLDQZ512rm:
    DestName = getRegName(MI->getOperand(0).getReg());
    if (MI->getOperand(NumOperands - 1).isImm())
      DecodePSLLDQMask(getRegOperandVectorVT(MI, MVT::i8, 0),
                       MI->getOperand(NumOperands - 1).getImm(),
                       ShuffleMask);
    break;

  case X86::PSRLDQri:
  case X86::VPSRLDQri:
  case X86::VPSRLDQYri:
  case X86::VPSRLDQZ128rr:
  case X86::VPSRLDQZ256rr:
  case X86::VPSRLDQZ512rr:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    LLVM_FALLTHROUGH;
  case X86::VPSRLDQZ128rm:
  case X86::VPSRLDQZ256rm:
  case X86::VPSRLDQZ512rm:
    DestName = getRegName(MI->getOperand(0).getReg());
    if (MI->getOperand(NumOperands - 1).isImm())
      DecodePSRLDQMask(getRegOperandVectorVT(MI, MVT::i8, 0),
                       MI->getOperand(NumOperands - 1).getImm(),
                       ShuffleMask);
    break;

  CASE_SHUF(PALIGNR, rri)
    Src1Name = getRegName(MI->getOperand(NumOperands - 2).getReg());
    RegForm = true;
    LLVM_FALLTHROUGH;

  CASE_SHUF(PALIGNR, rmi)
    Src2Name = getRegName(MI->getOperand(NumOperands-(RegForm?3:7)).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    if (MI->getOperand(NumOperands - 1).isImm())
      DecodePALIGNRMask(getRegOperandVectorVT(MI, MVT::i8, 0),
                        MI->getOperand(NumOperands - 1).getImm(),
                        ShuffleMask);
    break;

  CASE_AVX512_INS_COMMON(ALIGNQ, Z, rri)
  CASE_AVX512_INS_COMMON(ALIGNQ, Z256, rri)
  CASE_AVX512_INS_COMMON(ALIGNQ, Z128, rri)
    Src1Name = getRegName(MI->getOperand(NumOperands - 2).getReg());
    RegForm = true;
    LLVM_FALLTHROUGH;

  CASE_AVX512_INS_COMMON(ALIGNQ, Z, rmi)
  CASE_AVX512_INS_COMMON(ALIGNQ, Z256, rmi)
  CASE_AVX512_INS_COMMON(ALIGNQ, Z128, rmi)
    Src2Name = getRegName(MI->getOperand(NumOperands-(RegForm?3:7)).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    if (MI->getOperand(NumOperands - 1).isImm())
      DecodeVALIGNMask(getRegOperandVectorVT(MI, MVT::i64, 0),
                       MI->getOperand(NumOperands - 1).getImm(),
                       ShuffleMask);
    break;

  CASE_AVX512_INS_COMMON(ALIGND, Z, rri)
  CASE_AVX512_INS_COMMON(ALIGND, Z256, rri)
  CASE_AVX512_INS_COMMON(ALIGND, Z128, rri)
    Src1Name = getRegName(MI->getOperand(NumOperands - 2).getReg());
    RegForm = true;
    LLVM_FALLTHROUGH;

  CASE_AVX512_INS_COMMON(ALIGND, Z, rmi)
  CASE_AVX512_INS_COMMON(ALIGND, Z256, rmi)
  CASE_AVX512_INS_COMMON(ALIGND, Z128, rmi)
    Src2Name = getRegName(MI->getOperand(NumOperands-(RegForm?3:7)).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    if (MI->getOperand(NumOperands - 1).isImm())
      DecodeVALIGNMask(getRegOperandVectorVT(MI, MVT::i32, 0),
                       MI->getOperand(NumOperands - 1).getImm(),
                       ShuffleMask);
    break;

  CASE_SHUF(PSHUFD, ri)
    Src1Name = getRegName(MI->getOperand(NumOperands - 2).getReg());
    LLVM_FALLTHROUGH;

  CASE_SHUF(PSHUFD, mi)
    DestName = getRegName(MI->getOperand(0).getReg());
    if (MI->getOperand(NumOperands - 1).isImm())
      DecodePSHUFMask(getRegOperandVectorVT(MI, MVT::i32, 0),
                      MI->getOperand(NumOperands - 1).getImm(),
                      ShuffleMask);
    break;

  CASE_SHUF(PSHUFHW, ri)
    Src1Name = getRegName(MI->getOperand(NumOperands - 2).getReg());
    LLVM_FALLTHROUGH;

  CASE_SHUF(PSHUFHW, mi)
    DestName = getRegName(MI->getOperand(0).getReg());
    if (MI->getOperand(NumOperands - 1).isImm())
      DecodePSHUFHWMask(getRegOperandVectorVT(MI, MVT::i16, 0),
                        MI->getOperand(NumOperands - 1).getImm(),
                        ShuffleMask);
    break;

  CASE_SHUF(PSHUFLW, ri)
    Src1Name = getRegName(MI->getOperand(NumOperands - 2).getReg());
    LLVM_FALLTHROUGH;

  CASE_SHUF(PSHUFLW, mi)
    DestName = getRegName(MI->getOperand(0).getReg());
    if (MI->getOperand(NumOperands - 1).isImm())
      DecodePSHUFLWMask(getRegOperandVectorVT(MI, MVT::i16, 0),
                        MI->getOperand(NumOperands - 1).getImm(),
                        ShuffleMask);
    break;

  case X86::MMX_PSHUFWri:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    LLVM_FALLTHROUGH;

  case X86::MMX_PSHUFWmi:
    DestName = getRegName(MI->getOperand(0).getReg());
    if (MI->getOperand(NumOperands - 1).isImm())
      DecodePSHUFMask(MVT::v4i16,
                      MI->getOperand(NumOperands - 1).getImm(),
                      ShuffleMask);
    break;

  case X86::PSWAPDrr:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    LLVM_FALLTHROUGH;

  case X86::PSWAPDrm:
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodePSWAPMask(MVT::v2i32, ShuffleMask);
    break;

  CASE_UNPCK(PUNPCKHBW, r)
  case X86::MMX_PUNPCKHBWirr:
    Src2Name = getRegName(MI->getOperand(NumOperands - 1).getReg());
    RegForm = true;
    LLVM_FALLTHROUGH;

  CASE_UNPCK(PUNPCKHBW, m)
  case X86::MMX_PUNPCKHBWirm:
    Src1Name = getRegName(MI->getOperand(NumOperands-(RegForm?2:6)).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKHMask(getRegOperandVectorVT(MI, MVT::i8, 0), ShuffleMask);
    break;

  CASE_UNPCK(PUNPCKHWD, r)
  case X86::MMX_PUNPCKHWDirr:
    Src2Name = getRegName(MI->getOperand(NumOperands - 1).getReg());
    RegForm = true;
    LLVM_FALLTHROUGH;

  CASE_UNPCK(PUNPCKHWD, m)
  case X86::MMX_PUNPCKHWDirm:
    Src1Name = getRegName(MI->getOperand(NumOperands-(RegForm?2:6)).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKHMask(getRegOperandVectorVT(MI, MVT::i16, 0), ShuffleMask);
    break;

  CASE_UNPCK(PUNPCKHDQ, r)
  case X86::MMX_PUNPCKHDQirr:
    Src2Name = getRegName(MI->getOperand(NumOperands - 1).getReg());
    RegForm = true;
    LLVM_FALLTHROUGH;

  CASE_UNPCK(PUNPCKHDQ, m)
  case X86::MMX_PUNPCKHDQirm:
    Src1Name = getRegName(MI->getOperand(NumOperands-(RegForm?2:6)).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKHMask(getRegOperandVectorVT(MI, MVT::i32, 0), ShuffleMask);
    break;

  CASE_UNPCK(PUNPCKHQDQ, r)
    Src2Name = getRegName(MI->getOperand(NumOperands - 1).getReg());
    RegForm = true;
    LLVM_FALLTHROUGH;

  CASE_UNPCK(PUNPCKHQDQ, m)
    Src1Name = getRegName(MI->getOperand(NumOperands-(RegForm?2:6)).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKHMask(getRegOperandVectorVT(MI, MVT::i64, 0), ShuffleMask);
    break;

  CASE_UNPCK(PUNPCKLBW, r)
  case X86::MMX_PUNPCKLBWirr:
    Src2Name = getRegName(MI->getOperand(NumOperands - 1).getReg());
    RegForm = true;
    LLVM_FALLTHROUGH;

  CASE_UNPCK(PUNPCKLBW, m)
  case X86::MMX_PUNPCKLBWirm:
    Src1Name = getRegName(MI->getOperand(NumOperands-(RegForm?2:6)).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKLMask(getRegOperandVectorVT(MI, MVT::i8, 0), ShuffleMask);
    break;

  CASE_UNPCK(PUNPCKLWD, r)
  case X86::MMX_PUNPCKLWDirr:
    Src2Name = getRegName(MI->getOperand(NumOperands - 1).getReg());
    RegForm = true;
    LLVM_FALLTHROUGH;

  CASE_UNPCK(PUNPCKLWD, m)
  case X86::MMX_PUNPCKLWDirm:
    Src1Name = getRegName(MI->getOperand(NumOperands-(RegForm?2:6)).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKLMask(getRegOperandVectorVT(MI, MVT::i16, 0), ShuffleMask);
    break;

  CASE_UNPCK(PUNPCKLDQ, r)
  case X86::MMX_PUNPCKLDQirr:
    Src2Name = getRegName(MI->getOperand(NumOperands - 1).getReg());
    RegForm = true;
    LLVM_FALLTHROUGH;

  CASE_UNPCK(PUNPCKLDQ, m)
  case X86::MMX_PUNPCKLDQirm:
    Src1Name = getRegName(MI->getOperand(NumOperands-(RegForm?2:6)).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKLMask(getRegOperandVectorVT(MI, MVT::i32, 0), ShuffleMask);
    break;

  CASE_UNPCK(PUNPCKLQDQ, r)
    Src2Name = getRegName(MI->getOperand(NumOperands - 1).getReg());
    RegForm = true;
    LLVM_FALLTHROUGH;

  CASE_UNPCK(PUNPCKLQDQ, m)
    Src1Name = getRegName(MI->getOperand(NumOperands-(RegForm?2:6)).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeUNPCKLMask(getRegOperandVectorVT(MI, MVT::i64, 0), ShuffleMask);
    break;

  CASE_SHUF(SHUFPD, rri)
    Src2Name = getRegName(MI->getOperand(NumOperands - 2).getReg());
    RegForm = true;
    LLVM_FALLTHROUGH;

  CASE_SHUF(SHUFPD, rmi)
    if (MI->getOperand(NumOperands - 1).isImm())
      DecodeSHUFPMask(getRegOperandVectorVT(MI, MVT::f64, 0),
                      MI->getOperand(NumOperands - 1).getImm(),
                      ShuffleMask);
    Src1Name = getRegName(MI->getOperand(NumOperands-(RegForm?3:7)).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  CASE_SHUF(SHUFPS, rri)
    Src2Name = getRegName(MI->getOperand(NumOperands - 2).getReg());
    RegForm = true;
    LLVM_FALLTHROUGH;

  CASE_SHUF(SHUFPS, rmi)
    if (MI->getOperand(NumOperands - 1).isImm())
      DecodeSHUFPMask(getRegOperandVectorVT(MI, MVT::f32, 0),
                      MI->getOperand(NumOperands - 1).getImm(),
                      ShuffleMask);
    Src1Name = getRegName(MI->getOperand(NumOperands-(RegForm?3:7)).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  CASE_VSHUF(64X2, r)
    Src2Name = getRegName(MI->getOperand(NumOperands - 2).getReg());
    RegForm = true;
    LLVM_FALLTHROUGH;

  CASE_VSHUF(64X2, m)
    decodeVSHUF64x2FamilyMask(getRegOperandVectorVT(MI, MVT::i64, 0),
                              MI->getOperand(NumOperands - 1).getImm(),
                              ShuffleMask);
    Src1Name = getRegName(MI->getOperand(NumOperands-(RegForm?3:7)).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  CASE_VSHUF(32X4, r)
    Src2Name = getRegName(MI->getOperand(NumOperands - 2).getReg());
    RegForm = true;
    LLVM_FALLTHROUGH;

  CASE_VSHUF(32X4, m)
    decodeVSHUF64x2FamilyMask(getRegOperandVectorVT(MI, MVT::i32, 0),
                              MI->getOperand(NumOperands - 1).getImm(),
                              ShuffleMask);
    Src1Name = getRegName(MI->getOperand(NumOperands-(RegForm?3:7)).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  CASE_UNPCK(UNPCKLPD, r)
    Src2Name = getRegName(MI->getOperand(NumOperands - 1).getReg());
    RegForm = true;
    LLVM_FALLTHROUGH;

  CASE_UNPCK(UNPCKLPD, m)
    DecodeUNPCKLMask(getRegOperandVectorVT(MI, MVT::f64, 0), ShuffleMask);
    Src1Name = getRegName(MI->getOperand(NumOperands-(RegForm?2:6)).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  CASE_UNPCK(UNPCKLPS, r)
    Src2Name = getRegName(MI->getOperand(NumOperands - 1).getReg());
    RegForm = true;
    LLVM_FALLTHROUGH;

  CASE_UNPCK(UNPCKLPS, m)
    DecodeUNPCKLMask(getRegOperandVectorVT(MI, MVT::f32, 0), ShuffleMask);
    Src1Name = getRegName(MI->getOperand(NumOperands-(RegForm?2:6)).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  CASE_UNPCK(UNPCKHPD, r)
    Src2Name = getRegName(MI->getOperand(NumOperands - 1).getReg());
    RegForm = true;
    LLVM_FALLTHROUGH;

  CASE_UNPCK(UNPCKHPD, m)
    DecodeUNPCKHMask(getRegOperandVectorVT(MI, MVT::f64, 0), ShuffleMask);
    Src1Name = getRegName(MI->getOperand(NumOperands-(RegForm?2:6)).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  CASE_UNPCK(UNPCKHPS, r)
    Src2Name = getRegName(MI->getOperand(NumOperands - 1).getReg());
    RegForm = true;
    LLVM_FALLTHROUGH;

  CASE_UNPCK(UNPCKHPS, m)
    DecodeUNPCKHMask(getRegOperandVectorVT(MI, MVT::f32, 0), ShuffleMask);
    Src1Name = getRegName(MI->getOperand(NumOperands-(RegForm?2:6)).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  CASE_VPERMILPI(PERMILPS, r)
    Src1Name = getRegName(MI->getOperand(NumOperands - 2).getReg());
    LLVM_FALLTHROUGH;

  CASE_VPERMILPI(PERMILPS, m)
    if (MI->getOperand(NumOperands - 1).isImm())
      DecodePSHUFMask(getRegOperandVectorVT(MI, MVT::f32, 0),
                      MI->getOperand(NumOperands - 1).getImm(),
                      ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  CASE_VPERMILPI(PERMILPD, r)
    Src1Name = getRegName(MI->getOperand(NumOperands - 2).getReg());
    LLVM_FALLTHROUGH;

  CASE_VPERMILPI(PERMILPD, m)
    if (MI->getOperand(NumOperands - 1).isImm())
      DecodePSHUFMask(getRegOperandVectorVT(MI, MVT::f64, 0),
                      MI->getOperand(NumOperands - 1).getImm(),
                      ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  case X86::VPERM2F128rr:
  case X86::VPERM2I128rr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    LLVM_FALLTHROUGH;

  case X86::VPERM2F128rm:
  case X86::VPERM2I128rm:
    // For instruction comments purpose, assume the 256-bit vector is v4i64.
    if (MI->getOperand(NumOperands - 1).isImm())
      DecodeVPERM2X128Mask(MVT::v4i64,
                           MI->getOperand(NumOperands - 1).getImm(),
                           ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  CASE_VPERM(PERMPD, r)
    Src1Name = getRegName(MI->getOperand(NumOperands - 2).getReg());
    LLVM_FALLTHROUGH;

  CASE_VPERM(PERMPD, m)
    if (MI->getOperand(NumOperands - 1).isImm())
      DecodeVPERMMask(getRegOperandVectorVT(MI, MVT::f64, 0),
                      MI->getOperand(NumOperands - 1).getImm(),
                      ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  CASE_VPERM(PERMQ, r)
    Src1Name = getRegName(MI->getOperand(NumOperands - 2).getReg());
    LLVM_FALLTHROUGH;

  CASE_VPERM(PERMQ, m)
    if (MI->getOperand(NumOperands - 1).isImm())
      DecodeVPERMMask(getRegOperandVectorVT(MI, MVT::i64, 0),
                      MI->getOperand(NumOperands - 1).getImm(),
                      ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  case X86::MOVSDrr:
  case X86::VMOVSDrr:
  case X86::VMOVSDZrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    Src1Name = getRegName(MI->getOperand(1).getReg());
    LLVM_FALLTHROUGH;

  case X86::MOVSDrm:
  case X86::VMOVSDrm:
  case X86::VMOVSDZrm:
    DecodeScalarMoveMask(MVT::v2f64, nullptr == Src2Name, ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  case X86::MOVSSrr:
  case X86::VMOVSSrr:
  case X86::VMOVSSZrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    Src1Name = getRegName(MI->getOperand(1).getReg());
    LLVM_FALLTHROUGH;

  case X86::MOVSSrm:
  case X86::VMOVSSrm:
  case X86::VMOVSSZrm:
    DecodeScalarMoveMask(MVT::v4f32, nullptr == Src2Name, ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  case X86::MOVPQI2QIrr:
  case X86::MOVZPQILo2PQIrr:
  case X86::VMOVPQI2QIrr:
  case X86::VMOVZPQILo2PQIrr:
  case X86::VMOVZPQILo2PQIZrr:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    LLVM_FALLTHROUGH;

  case X86::MOVQI2PQIrm:
  case X86::VMOVQI2PQIrm:
  case X86::VMOVQI2PQIZrm:
    DecodeZeroMoveLowMask(MVT::v2i64, ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  case X86::MOVDI2PDIrm:
  case X86::VMOVDI2PDIrm:
  case X86::VMOVDI2PDIZrm:
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

  case X86::VBROADCASTF128:
  case X86::VBROADCASTI128:
  CASE_AVX512_INS_COMMON(BROADCASTF64X2, Z128, rm)
  CASE_AVX512_INS_COMMON(BROADCASTI64X2, Z128, rm)
    DecodeSubVectorBroadcast(MVT::v4f64, MVT::v2f64, ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  CASE_AVX512_INS_COMMON(BROADCASTF64X2, , rm)
  CASE_AVX512_INS_COMMON(BROADCASTI64X2, , rm)
    DecodeSubVectorBroadcast(MVT::v8f64, MVT::v2f64, ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  CASE_AVX512_INS_COMMON(BROADCASTF64X4, , rm)
  CASE_AVX512_INS_COMMON(BROADCASTI64X4, , rm)
    DecodeSubVectorBroadcast(MVT::v8f64, MVT::v4f64, ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  CASE_AVX512_INS_COMMON(BROADCASTF32X4, Z256, rm)
  CASE_AVX512_INS_COMMON(BROADCASTI32X4, Z256, rm)
    DecodeSubVectorBroadcast(MVT::v8f32, MVT::v4f32, ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  CASE_AVX512_INS_COMMON(BROADCASTF32X4, , rm)
  CASE_AVX512_INS_COMMON(BROADCASTI32X4, , rm)
    DecodeSubVectorBroadcast(MVT::v16f32, MVT::v4f32, ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  CASE_AVX512_INS_COMMON(BROADCASTF32X8, , rm)
  CASE_AVX512_INS_COMMON(BROADCASTI32X8, , rm)
    DecodeSubVectorBroadcast(MVT::v16f32, MVT::v8f32, ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  CASE_AVX512_INS_COMMON(BROADCASTF32X2, Z256, r)
  CASE_AVX512_INS_COMMON(BROADCASTI32X2, Z256, r)
    Src1Name = getRegName(MI->getOperand(NumOperands - 1).getReg());
    LLVM_FALLTHROUGH;
  CASE_AVX512_INS_COMMON(BROADCASTF32X2, Z256, m)
  CASE_AVX512_INS_COMMON(BROADCASTI32X2, Z256, m)
    DecodeSubVectorBroadcast(MVT::v8f32, MVT::v2f32, ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  CASE_AVX512_INS_COMMON(BROADCASTF32X2, Z, r)
  CASE_AVX512_INS_COMMON(BROADCASTI32X2, Z, r)
    Src1Name = getRegName(MI->getOperand(NumOperands - 1).getReg());
    LLVM_FALLTHROUGH;
  CASE_AVX512_INS_COMMON(BROADCASTF32X2, Z, m)
  CASE_AVX512_INS_COMMON(BROADCASTI32X2, Z, m)
    DecodeSubVectorBroadcast(MVT::v16f32, MVT::v2f32, ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  CASE_PMOVZX(PMOVZXBW, r)
  CASE_PMOVZX(PMOVZXBD, r)
  CASE_PMOVZX(PMOVZXBQ, r)
    Src1Name = getRegName(MI->getOperand(NumOperands - 1).getReg());
    LLVM_FALLTHROUGH;

  CASE_PMOVZX(PMOVZXBW, m)
  CASE_PMOVZX(PMOVZXBD, m)
  CASE_PMOVZX(PMOVZXBQ, m)
    DecodeZeroExtendMask(MVT::i8, getZeroExtensionResultType(MI), ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  CASE_PMOVZX(PMOVZXWD, r)
  CASE_PMOVZX(PMOVZXWQ, r)
    Src1Name = getRegName(MI->getOperand(NumOperands - 1).getReg());
    LLVM_FALLTHROUGH;

  CASE_PMOVZX(PMOVZXWD, m)
  CASE_PMOVZX(PMOVZXWQ, m)
    DecodeZeroExtendMask(MVT::i16, getZeroExtensionResultType(MI), ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  CASE_PMOVZX(PMOVZXDQ, r)
    Src1Name = getRegName(MI->getOperand(NumOperands - 1).getReg());
    LLVM_FALLTHROUGH;

  CASE_PMOVZX(PMOVZXDQ, m)
    DecodeZeroExtendMask(MVT::i32, getZeroExtensionResultType(MI), ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  }

  // The only comments we decode are shuffles, so give up if we were unable to
  // decode a shuffle mask.
  if (ShuffleMask.empty())
    return false;

  if (!DestName) DestName = Src1Name;
  OS << (DestName ? getMaskName(MI, DestName, getRegName) : "mem") << " = ";

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

  // We successfully added a comment to this instruction.
  return true;
}
