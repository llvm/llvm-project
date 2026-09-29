//===- LoongArchMatInt.cpp - Immediate materialisation ---------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LoongArchMatInt.h"
#include "MCTargetDesc/LoongArchMCTargetDesc.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

LoongArchMatInt::InstSeq LoongArchMatInt::generateInstSeq(int64_t Val) {
  // Val:
  // |            hi32              |              lo32            |
  // +-----------+------------------+------------------+-----------+
  // | Highest12 |    Higher20      |       Hi20       |    Lo12   |
  // +-----------+------------------+------------------+-----------+
  // 63        52 51              32 31              12 11         0
  //
  const int64_t Highest12 = Val >> 52 & 0xFFF;
  const int64_t Higher20 = Val >> 32 & 0xFFFFF;
  const int64_t Hi20 = Val >> 12 & 0xFFFFF;
  const int64_t Lo12 = Val & 0xFFF;
  InstSeq Insts;

  // LU52I_D used for: Bits[63:52] | Bits[51:0].
  if (Highest12 != 0 && SignExtend64<52>(Val) == 0) {
    Insts.push_back(Inst(LoongArch::LU52I_D, SignExtend64<12>(Highest12)));
    return Insts;
  }

  // lo32
  if (Hi20 == 0)
    Insts.push_back(Inst(LoongArch::ORI, Lo12));
  else if (SignExtend32<1>(Lo12 >> 11) == SignExtend32<20>(Hi20))
    Insts.push_back(Inst(LoongArch::ADDI_W, SignExtend64<12>(Lo12)));
  else {
    Insts.push_back(Inst(LoongArch::LU12I_W, SignExtend64<20>(Hi20)));
    if (Lo12 != 0)
      Insts.push_back(Inst(LoongArch::ORI, Lo12));
  }

  // hi32
  // Higher20
  if (SignExtend32<1>(Hi20 >> 19) != SignExtend32<20>(Higher20))
    Insts.push_back(Inst(LoongArch::LU32I_D, SignExtend64<20>(Higher20)));

  // Highest12
  if (SignExtend32<1>(Higher20 >> 19) != SignExtend32<12>(Highest12))
    Insts.push_back(Inst(LoongArch::LU52I_D, SignExtend64<12>(Highest12)));

  size_t N = Insts.size();
  if (N < 3)
    return Insts;

  // When the number of instruction sequences is greater than 2, we have the
  // opportunity to optimize using the BSTRINS_D instruction. The scenario is as
  // follows:
  //
  // N of Insts = 3
  // 1. ORI + LU32I_D + LU52I_D     =>     ORI + BSTRINS_D, TmpVal = ORI
  // 2. ADDI_W + LU32I_D + LU52I_D  =>  ADDI_W + BSTRINS_D, TmpVal = ADDI_W
  // 3. LU12I_W + ORI + LU32I_D     =>     ORI + BSTRINS_D, TmpVal = ORI
  // 4. LU12I_W + LU32I_D + LU52I_D => LU12I_W + BSTRINS_D, TmpVal = LU12I_W
  //
  // N of Insts = 4
  // 5. LU12I_W + ORI + LU32I_D + LU52I_D => LU12I_W + ORI + BSTRINS_D
  //                                      => ORI + LU52I_D + BSTRINS_D
  //    TmpVal = (LU12I_W | ORI) or (ORI | LU52I_D)
  // The BSTRINS_D instruction will use the `TmpVal` to construct the `Val`.
  uint64_t TmpVal1 = 0;
  uint64_t TmpVal2 = 0;
  switch (Insts[0].Opc) {
  default:
    llvm_unreachable("unexpected opcode");
    break;
  case LoongArch::LU12I_W:
    if (Insts[1].Opc == LoongArch::ORI) {
      TmpVal1 = Insts[1].Imm;
      if (N == 3)
        break;
      TmpVal2 = static_cast<uint64_t>(Insts[3].Imm) << 52 | TmpVal1;
    }
    TmpVal1 |= static_cast<uint64_t>(Insts[0].Imm) << 12;
    break;
  case LoongArch::ORI:
  case LoongArch::ADDI_W:
    TmpVal1 = Insts[0].Imm;
    break;
  }

  uint64_t Msb = 32;
  uint64_t HighMask = ~((1ULL << (Msb + 1)) - 1);
  for (; Msb < 64; ++Msb, HighMask = HighMask << 1) {
    for (uint64_t Lsb = Msb; Lsb > 0; --Lsb) {
      uint64_t LowMask = (1ULL << Lsb) - 1;
      uint64_t Mask = HighMask | LowMask;
      uint64_t LsbToZero = TmpVal1 & ((1ULL << (Msb - Lsb + 1)) - 1);
      uint64_t MsbToLsb = LsbToZero << Lsb;
      if ((MsbToLsb | (TmpVal1 & Mask)) == (uint64_t)Val) {
        if (Insts[1].Opc == LoongArch::ORI && N == 3)
          Insts[0] = Insts[1];
        Insts.pop_back_n(2);
        Insts.push_back(Inst(LoongArch::BSTRINS_D, Msb << 32 | Lsb));
        return Insts;
      }
      if (TmpVal2 != 0) {
        LsbToZero = TmpVal2 & ((1ULL << (Msb - Lsb + 1)) - 1);
        MsbToLsb = LsbToZero << Lsb;
        if ((MsbToLsb | (TmpVal2 & Mask)) == (uint64_t)Val) {
          Insts[0] = Insts[1];
          Insts[1] = Insts[3];
          Insts.pop_back_n(2);
          Insts.push_back(Inst(LoongArch::BSTRINS_D, Msb << 32 | Lsb));
          return Insts;
        }
      }
    }
  }

  return Insts;
}
