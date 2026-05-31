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

def : IntrinsicRename<"llvm.sin.f32", "air.fast_sin.f32">;
def : IntrinsicRename<"llvm.cos.f32", "air.fast_cos.f32">;
def : IntrinsicRename<"llvm.sin.f16", "air.fast_sin.f16">;
def : IntrinsicRename<"llvm.cos.f16", "air.fast_cos.f16">;

def : IntrinsicRename<"llvm.exp.f32",  "air.fast_exp.f32">;
def : IntrinsicRename<"llvm.log.f32",  "air.fast_log.f32">;
def : IntrinsicRename<"llvm.exp2.f32", "air.fast_exp2.f32">;
def : IntrinsicRename<"llvm.log2.f32", "air.fast_log2.f32">;
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
