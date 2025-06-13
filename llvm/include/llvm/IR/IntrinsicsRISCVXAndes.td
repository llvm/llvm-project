//===- IntrinsicsRISCVXAndes.td - Andes intrinsics ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the Andes vendor intrinsics for RISC-V.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "riscv" in {
  // Andes Vector Packed FP16 Extension
  defm nds_vfpmadt : RISCVBinaryAAXRoundingMode;
  defm nds_vfpmadb : RISCVBinaryAAXRoundingMode;

  // Andes Vector Dot Product Extension
  defm nds_vd4dots  : RISCVTernaryWide;
  defm nds_vd4dotu  : RISCVTernaryWide;
  defm nds_vd4dotsu : RISCVTernaryWide;
}
