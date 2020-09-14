//===-- LLVMAVX512.td - LLVMAVX512 dialect op definitions --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the LLVMAVX512 dialect.
//
//===----------------------------------------------------------------------===//

#ifndef LLVMIR_AVX512_OPS
#define LLVMIR_AVX512_OPS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

//===----------------------------------------------------------------------===//
// LLVMAVX512 dialect definition
//===----------------------------------------------------------------------===//

def LLVMAVX512_Dialect : Dialect {
  let name = "llvm_avx512";
  let cppNamespace = "::mlir::LLVM";
}

//----------------------------------------------------------------------------//
// MLIR LLVM AVX512 intrinsics using the MLIR LLVM Dialect type system
//----------------------------------------------------------------------------//

class LLVMAVX512_IntrOp<string mnemonic, list<OpTrait> traits = []> :
  LLVM_IntrOpBase<LLVMAVX512_Dialect, mnemonic,
                  "x86_avx512_" # !subst(".", "_", mnemonic),
                  [], [], traits, 1>;

def LLVM_x86_avx512_mask_rndscale_ps_512 :
  LLVMAVX512_IntrOp<"mask.rndscale.ps.512">,
  Arguments<(ins LLVM_Type, LLVM_Type, LLVM_Type, LLVM_Type, LLVM_Type)>;

def LLVM_x86_avx512_mask_rndscale_pd_512 :
  LLVMAVX512_IntrOp<"mask.rndscale.pd.512">,
  Arguments<(ins LLVM_Type, LLVM_Type, LLVM_Type, LLVM_Type, LLVM_Type)>;

def LLVM_x86_avx512_mask_scalef_ps_512 :
  LLVMAVX512_IntrOp<"mask.scalef.ps.512">,
  Arguments<(ins LLVM_Type, LLVM_Type, LLVM_Type, LLVM_Type, LLVM_Type)>;

def LLVM_x86_avx512_mask_scalef_pd_512 :
  LLVMAVX512_IntrOp<"mask.scalef.pd.512">,
  Arguments<(ins LLVM_Type, LLVM_Type, LLVM_Type, LLVM_Type, LLVM_Type)>;

#endif // AVX512_OPS
