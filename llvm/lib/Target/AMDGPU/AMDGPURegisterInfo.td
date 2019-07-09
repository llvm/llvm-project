//===-- AMDGPURegisterInfo.td - AMDGPU register info -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tablegen register definitions common to all hw codegen targets.
//
//===----------------------------------------------------------------------===//

let Namespace = "AMDGPU" in {

foreach Index = 0-31 in {
  def sub#Index : SubRegIndex<32, !shl(Index, 5)>;
}

}

include "SIRegisterInfo.td"
