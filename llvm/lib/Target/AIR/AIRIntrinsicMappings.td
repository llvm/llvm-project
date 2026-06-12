//===- AIRIntrinsicMappings.td ------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Mapping of LLVM intrinsic names to their Apple AIR equivalents. The records
// declared here are consumed by -gen-searchable-tables to produce a binary-
// searchable table used by the LLVMToAIRIntrinsics pass.
//
//===----------------------------------------------------------------------===//

include "llvm/TableGen/SearchableTable.td"

class IntrinsicRename<string llvm, string air> {
  string LLVMName = llvm;
  string AIRName = air;
}

def AIRIntrinsicRenames : GenericTable {
  let FilterClass = "IntrinsicRename";
  let Fields = ["LLVMName", "AIRName"];
}

// LLVM math intrinsic -> AIR intrinsic name mappings.

def : IntrinsicRename<"llvm.maxnum.f32", "air.fmax.f32">;
def : IntrinsicRename<"llvm.minnum.f32", "air.fmin.f32">;
def : IntrinsicRename<"llvm.maxnum.f16", "air.fmax.f16">;
def : IntrinsicRename<"llvm.minnum.f16", "air.fmin.f16">;
def : IntrinsicRename<"llvm.maximum.f32", "air.fmax.f32">;
def : IntrinsicRename<"llvm.minimum.f32", "air.fmin.f32">;
def : IntrinsicRename<"llvm.maximum.f16", "air.fmax.f16">;
def : IntrinsicRename<"llvm.minimum.f16", "air.fmin.f16">;
def : IntrinsicRename<"llvm.maximum.v2f32", "air.fmax.v2f32">;
def : IntrinsicRename<"llvm.maximum.v3f32", "air.fmax.v3f32">;
def : IntrinsicRename<"llvm.maximum.v4f32", "air.fmax.v4f32">;
def : IntrinsicRename<"llvm.minimum.v2f32", "air.fmin.v2f32">;
def : IntrinsicRename<"llvm.minimum.v3f32", "air.fmin.v3f32">;
def : IntrinsicRename<"llvm.minimum.v4f32", "air.fmin.v4f32">;
def : IntrinsicRename<"llvm.maximum.v2f16", "air.fmax.v2f16">;
def : IntrinsicRename<"llvm.maximum.v4f16", "air.fmax.v4f16">;
def : IntrinsicRename<"llvm.minimum.v2f16", "air.fmin.v2f16">;
def : IntrinsicRename<"llvm.minimum.v4f16", "air.fmin.v4f16">;

def : IntrinsicRename<"llvm.sin.f32", "air.sin.f32">;
def : IntrinsicRename<"llvm.cos.f32", "air.cos.f32">;
def : IntrinsicRename<"llvm.sin.f16", "air.sin.f16">;
def : IntrinsicRename<"llvm.cos.f16", "air.cos.f16">;

def : IntrinsicRename<"llvm.exp.f32",  "air.fast_exp.f32">;
def : IntrinsicRename<"llvm.log.f32",  "air.fast_log.f32">;
def : IntrinsicRename<"llvm.exp2.f32", "air.exp2.f32">;
def : IntrinsicRename<"llvm.log2.f32", "air.log2.f32">;
def : IntrinsicRename<"llvm.pow.f32",  "air.pow.f32">;
def : IntrinsicRename<"llvm.exp.f16",  "air.fast_exp.f16">;
def : IntrinsicRename<"llvm.log.f16",  "air.fast_log.f16">;
def : IntrinsicRename<"llvm.exp2.f16", "air.fast_exp2.f16">;
def : IntrinsicRename<"llvm.log2.f16", "air.fast_log2.f16">;

def : IntrinsicRename<"llvm.sqrt.f32",  "air.fast_sqrt.f32">;
def : IntrinsicRename<"llvm.fabs.f32",  "air.fabs.f32">;
def : IntrinsicRename<"llvm.floor.f32", "air.fast_floor.f32">;
def : IntrinsicRename<"llvm.ceil.f32",  "air.fast_ceil.f32">;
def : IntrinsicRename<"llvm.sqrt.f16",  "air.fast_sqrt.f16">;
def : IntrinsicRename<"llvm.fabs.f16",  "air.fabs.f16">;
def : IntrinsicRename<"llvm.floor.f16", "air.fast_floor.f16">;
def : IntrinsicRename<"llvm.ceil.f16",  "air.fast_ceil.f16">;

def : IntrinsicRename<"llvm.fma.f32", "air.fma.f32">;
def : IntrinsicRename<"llvm.fma.f16", "air.fma.f16">;

def : IntrinsicRename<"llvm.rint.f32", "air.fast_rint.f32">;
def : IntrinsicRename<"llvm.rint.f16", "air.fast_rint.f16">;

def : IntrinsicRename<"llvm.maxnum.v2f32", "air.fast_fmax.v2f32">;
def : IntrinsicRename<"llvm.maxnum.v3f32", "air.fast_fmax.v3f32">;
def : IntrinsicRename<"llvm.maxnum.v4f32", "air.fast_fmax.v4f32">;
def : IntrinsicRename<"llvm.minnum.v2f32", "air.fast_fmin.v2f32">;
def : IntrinsicRename<"llvm.minnum.v3f32", "air.fast_fmin.v3f32">;
def : IntrinsicRename<"llvm.minnum.v4f32", "air.fast_fmin.v4f32">;
def : IntrinsicRename<"llvm.sin.v2f32", "air.sin.v2f32">;
def : IntrinsicRename<"llvm.sin.v3f32", "air.sin.v3f32">;
def : IntrinsicRename<"llvm.sin.v4f32", "air.sin.v4f32">;
def : IntrinsicRename<"llvm.cos.v2f32", "air.cos.v2f32">;
def : IntrinsicRename<"llvm.cos.v3f32", "air.cos.v3f32">;
def : IntrinsicRename<"llvm.cos.v4f32", "air.cos.v4f32">;
def : IntrinsicRename<"llvm.exp.v2f32", "air.fast_exp.v2f32">;
def : IntrinsicRename<"llvm.exp.v3f32", "air.fast_exp.v3f32">;
def : IntrinsicRename<"llvm.exp.v4f32", "air.fast_exp.v4f32">;
def : IntrinsicRename<"llvm.log.v2f32", "air.fast_log.v2f32">;
def : IntrinsicRename<"llvm.log.v3f32", "air.fast_log.v3f32">;
def : IntrinsicRename<"llvm.log.v4f32", "air.fast_log.v4f32">;
def : IntrinsicRename<"llvm.exp2.v2f32", "air.exp2.v2f32">;
def : IntrinsicRename<"llvm.exp2.v3f32", "air.exp2.v3f32">;
def : IntrinsicRename<"llvm.exp2.v4f32", "air.exp2.v4f32">;
def : IntrinsicRename<"llvm.log2.v2f32", "air.log2.v2f32">;
def : IntrinsicRename<"llvm.log2.v3f32", "air.log2.v3f32">;
def : IntrinsicRename<"llvm.log2.v4f32", "air.log2.v4f32">;
def : IntrinsicRename<"llvm.pow.v2f32", "air.pow.v2f32">;
def : IntrinsicRename<"llvm.pow.v3f32", "air.pow.v3f32">;
def : IntrinsicRename<"llvm.pow.v4f32", "air.pow.v4f32">;
def : IntrinsicRename<"llvm.sqrt.v2f32", "air.fast_sqrt.v2f32">;
def : IntrinsicRename<"llvm.sqrt.v3f32", "air.fast_sqrt.v3f32">;
def : IntrinsicRename<"llvm.sqrt.v4f32", "air.fast_sqrt.v4f32">;
def : IntrinsicRename<"llvm.fabs.v2f32", "air.fast_fabs.v2f32">;
def : IntrinsicRename<"llvm.fabs.v3f32", "air.fast_fabs.v3f32">;
def : IntrinsicRename<"llvm.fabs.v4f32", "air.fast_fabs.v4f32">;
def : IntrinsicRename<"llvm.floor.v2f32", "air.fast_floor.v2f32">;
def : IntrinsicRename<"llvm.floor.v3f32", "air.fast_floor.v3f32">;
def : IntrinsicRename<"llvm.floor.v4f32", "air.fast_floor.v4f32">;
def : IntrinsicRename<"llvm.ceil.v2f32", "air.fast_ceil.v2f32">;
def : IntrinsicRename<"llvm.ceil.v3f32", "air.fast_ceil.v3f32">;
def : IntrinsicRename<"llvm.ceil.v4f32", "air.fast_ceil.v4f32">;
def : IntrinsicRename<"llvm.rint.v2f32", "air.fast_rint.v2f32">;
def : IntrinsicRename<"llvm.rint.v3f32", "air.fast_rint.v3f32">;
def : IntrinsicRename<"llvm.rint.v4f32", "air.fast_rint.v4f32">;
def : IntrinsicRename<"llvm.fma.v2f32", "air.fma.v2f32">;
def : IntrinsicRename<"llvm.fma.v3f32", "air.fma.v3f32">;
def : IntrinsicRename<"llvm.fma.v4f32", "air.fma.v4f32">;
