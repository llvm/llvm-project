//===- AttributesAMDGPU.td - Defines AMDGPU attributes -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines AMDGPU specific attributes.
//
//===----------------------------------------------------------------------===//

def AMDGPUUnsafeFPAtomics : StrBoolAttr<"amdgpu-unsafe-fp-atomics">;
def : MergeRule<"setAND<AMDGPUUnsafeFPAtomicsAttr>">;
