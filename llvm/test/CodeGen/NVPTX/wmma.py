# This test generates all variants of wmma intrinsics and verifies that LLVM
# generates correct instructions for them.

# Check all variants of instructions supported by PTX60 on SM70
# RUN: %python %s --ptx=60 --gpu-arch=70 > %t-ptx60-sm_70.ll
# RUN: FileCheck %t-ptx60-sm_70.ll < %t-ptx60-sm_70.ll \
# RUN:           --check-prefixes=INTRINSICS,M16N16
# RUN: FileCheck %t-ptx60-sm_70.ll < %t-ptx60-sm_70.ll \
# RUN:           --check-prefixes=INTRINSICS,NOEXTGEOM,NOINT,NOSUBINT,NOMMA,NODOUBLE,NOALTFLOAT,NOLDMATRIX
# RUN: llc < %t-ptx60-sm_70.ll -march=nvptx64 -mcpu=sm_70 -mattr=+ptx60 \
# RUN:           | FileCheck %t-ptx60-sm_70.ll
# RUN: %if ptxas %{                                                       \
# RUN:   llc < %t-ptx60-sm_70.ll -march=nvptx64 -mcpu=sm_70 -mattr=+ptx60 \
# RUN:           | %ptxas-verify -arch=sm_70                              \
# RUN: %}

# Check all variants of instructions supported by PTX61 on SM70
# RUN: %python %s --ptx=61 --gpu-arch=70 > %t-ptx61-sm_70.ll
# RUN: FileCheck %t-ptx61-sm_70.ll < %t-ptx61-sm_70.ll \
# RUN:           --check-prefixes=INTRINSICS,M16N16,EXTGEOM
# RUN: FileCheck %t-ptx61-sm_70.ll < %t-ptx61-sm_70.ll \
# RUN:           --check-prefixes=INTRINSICS,NOINT,NOSUBINT,NOMMA,NODOUBLE,NOALTFLOAT,NOLDMATRIX
# RUN: llc < %t-ptx61-sm_70.ll -march=nvptx64 -mcpu=sm_70 -mattr=+ptx61 \
# RUN:           | FileCheck %t-ptx61-sm_70.ll
# RUN: %if ptxas-9.1 %{                                                   \
# RUN:   llc < %t-ptx61-sm_70.ll -march=nvptx64 -mcpu=sm_70 -mattr=+ptx61 \
# RUN:           | %ptxas-verify -arch=sm_70                              \
# RUN: %}

# Check all variants of instructions supported by PTX63 on SM72
# RUN: %python %s --ptx=63 --gpu-arch=72 > %t-ptx63-sm_72.ll
# RUN: FileCheck %t-ptx63-sm_72.ll < %t-ptx63-sm_72.ll \
# RUN:           --check-prefixes=INTRINSICS,M16N16,EXTGEOM,INT
# RUN: FileCheck %t-ptx63-sm_72.ll < %t-ptx63-sm_72.ll \
# RUN:           --check-prefixes=INTRINSICS,NOSUBINT,NOMMA,NODOUBLE,NOALTFLOAT,NOLDMATRIX
# RUN: llc < %t-ptx63-sm_72.ll -march=nvptx64 -mcpu=sm_72 -mattr=+ptx63 \
# RUN:           | FileCheck %t-ptx63-sm_72.ll
# RUN: %if ptxas-10.0 %{                                                  \
# RUN:   llc < %t-ptx63-sm_72.ll -march=nvptx64 -mcpu=sm_72 -mattr=+ptx63 \
# RUN:           | %ptxas-verify -arch=sm_72                              \
# RUN: %}

# Check all variants of instructions supported by PTX63 on SM75
# RUN: %python %s --ptx=63 --gpu-arch=75 > %t-ptx63-sm_75.ll
# RUN: FileCheck %t-ptx63-sm_75.ll < %t-ptx63-sm_75.ll \
# RUN:           --check-prefixes=INTRINSICS,M16N16,EXTGEOM,INT,SUBINT
# RUN: FileCheck %t-ptx63-sm_75.ll < %t-ptx63-sm_75.ll \
# RUN:           --check-prefixes=INTRINSICS,NOMMA,NODOUBLE,NOALTFLOAT,NOLDMATRIX
# RUN: llc < %t-ptx63-sm_75.ll -march=nvptx64 -mcpu=sm_75 -mattr=+ptx63 \
# RUN:           | FileCheck %t-ptx63-sm_75.ll
# RUN: %if ptxas-10.0 %{                                                  \
# RUN:   llc < %t-ptx63-sm_75.ll -march=nvptx64 -mcpu=sm_75 -mattr=+ptx63 \
# RUN:           | %ptxas-verify -arch=sm_75                              \
# RUN: %}

# Check all variants of instructions supported by PTX64 on SM70+
# RUN: %python %s --ptx=64 --gpu-arch=70 > %t-ptx64-sm_70.ll
# RUN: FileCheck %t-ptx64-sm_70.ll < %t-ptx64-sm_70.ll \
# RUN:           --check-prefixes=INTRINSICS,M16N16,EXTGEOM,MMA
# RUN: FileCheck %t-ptx64-sm_70.ll < %t-ptx64-sm_70.ll \
# RUN:           --check-prefixes=INTRINSICS,NOINT,NOSUBINT,NODOUBLE,NOALTFLOAT,NOLDMATRIX
# RUN: llc < %t-ptx64-sm_70.ll -march=nvptx64 -mcpu=sm_70 -mattr=+ptx64 \
# RUN:           | FileCheck %t-ptx64-sm_70.ll
# RUN: %if ptxas-10.1 %{                                                  \
# RUN:   llc < %t-ptx64-sm_70.ll -march=nvptx64 -mcpu=sm_70 -mattr=+ptx64 \
# RUN:           | %ptxas-verify -arch=sm_70                              \
# RUN: %}

# Check all variants of instructions supported by PTX65 on SM75+
# RUN: %python %s --ptx=65 --gpu-arch=75 > %t-ptx65-sm_75.ll
# RUN: FileCheck %t-ptx65-sm_75.ll < %t-ptx65-sm_75.ll \
# RUN:           --check-prefixes=INTRINSICS,M16N16,EXTGEOM,INT,SUBINT,MMA,PTX65MMA,PTX65LDMATRIX
# RUN: FileCheck %t-ptx65-sm_75.ll < %t-ptx65-sm_75.ll \
# RUN:           --check-prefixes=INTRINSICS
# RUN: llc < %t-ptx65-sm_75.ll -march=nvptx64 -mcpu=sm_75 -mattr=+ptx65 \
# RUN:           | FileCheck %t-ptx65-sm_75.ll
# RUN: %if ptxas-10.2 %{                                                  \
# RUN:   llc < %t-ptx65-sm_75.ll -march=nvptx64 -mcpu=sm_75 -mattr=+ptx65 \
# RUN:           | %ptxas-verify -arch=sm_75                              \
# RUN: %}

# Check all variants of instructions supported by PTX71 on SM80+
# RUN: %python %s --ptx=71 --gpu-arch=80 > %t-ptx71-sm_80.ll
# RUN: FileCheck %t-ptx71-sm_80.ll < %t-ptx71-sm_80.ll \
# RUN:           --check-prefixes=INTRINSICS,M16N16,EXTGEOM,INT,SUBINT,MMA,ALTFLOAT,DOUBLE,PTX65MMA,PTX65LDMATRIX,PTX71MMA
# RUN: FileCheck %t-ptx71-sm_80.ll < %t-ptx71-sm_80.ll \
# RUN:           --check-prefixes=INTRINSICS
# RUN: llc < %t-ptx71-sm_80.ll -march=nvptx64 -mcpu=sm_80 -mattr=+ptx71 \
# RUN:           | FileCheck %t-ptx71-sm_80.ll
# RUN: %if ptxas-11.1 %{                                                  \
# RUN:   llc < %t-ptx71-sm_80.ll -march=nvptx64 -mcpu=sm_80 -mattr=+ptx71 \
# RUN:           | %ptxas-verify -arch=sm_80                              \
# RUN: %}

from __future__ import print_function

import argparse
from itertools import product
from string import Template

class MMAType:
  def __init__(self, ptx_type):
    self.ptx_type = ptx_type
    self.llvm_type = {
        "f16"  : "<2 x half>",
        "f32"  : "float",
        "f64"  : "double",
        "s32"  : "i32",
        "b16"  : "i32",
        "s8"   : "i32",
        "u8"   : "i32",
        "s4"   : "i32",
        "u4"   : "i32",
        "b1"   : "i32",
        "bf16" : "i32",
        "tf32" : "i32",
    }[ptx_type];

    self.ptx_reg_pattern = {
        "f16" : "%hh[0-9]+",
        "f32" : "%f[0-9]+",
        "f64" : "%fd[0-9]+",
    }.get(ptx_type, "%r[0-9]+")

  def __repr__(self):
    return "%s/%s" % (self.ptx_type, self.llvm_type)

class MMAFrag:
  def __init__(self, geom, frag, ptx_elt_type):
    self.geom = geom
    self.frag = frag
    self.mma_type = MMAType(ptx_elt_type);
    self.nregs = {
        # u8/s8 -> s32 @ m16n16k16/m8n32k16/m32n8k16
        "m16n16k16:a:u8" : 2,
        "m16n16k16:a:s8" : 2,
        "m16n16k16:b:u8" : 2,
        "m16n16k16:b:s8" : 2,
        "m16n16k16:c:s32" : 8,
        "m16n16k16:d:s32" : 8,

        "m8n32k16:a:u8" : 1,
        "m8n32k16:a:s8" : 1,
        "m8n32k16:b:u8" : 4,
        "m8n32k16:b:s8" : 4,
        "m8n32k16:c:s32" : 8,
        "m8n32k16:d:s32" : 8,

        "m32n8k16:a:u8" : 4,
        "m32n8k16:a:s8" : 4,
        "m32n8k16:b:u8" : 1,
        "m32n8k16:b:s8" : 1,
        "m32n8k16:c:s32" : 8,
        "m32n8k16:d:s32" : 8,

        "m8n8k16:a:u8": 1,
        "m8n8k16:a:s8": 1,
        "m8n8k16:b:u8": 1,
        "m8n8k16:b:s8": 1,
        "m8n8k16:c:s32": 2,
        "m8n8k16:d:s32": 2,

        "m16n8k16:a:u8": 2,
        "m16n8k16:a:s8": 2,
        "m16n8k16:b:u8": 1,
        "m16n8k16:b:s8": 1,
        "m16n8k16:c:s32": 4,
        "m16n8k16:d:s32": 4,

        "m16n8k32:a:u8": 4,
        "m16n8k32:a:s8": 4,
        "m16n8k32:b:u8": 2,
        "m16n8k32:b:s8": 2,
        "m16n8k32:c:s32": 4,
        "m16n8k32:d:s32": 4,

        # u4/s4 -> s32 @ m8n8k32 (u4/s4)
        "m8n8k32:a:u4" : 1,
        "m8n8k32:a:s4" : 1,
        "m8n8k32:b:u4" : 1,
        "m8n8k32:b:s4" : 1,
        "m8n8k32:c:s32" : 2,
        "m8n8k32:d:s32" : 2,

        "m16n8k32:a:u4" : 2,
        "m16n8k32:a:s4" : 2,
        "m16n8k32:b:u4" : 1,
        "m16n8k32:b:s4" : 1,
        "m16n8k32:c:s32" : 4,
        "m16n8k32:d:s32" : 4,

        "m16n8k64:a:u4" : 4,
        "m16n8k64:a:s4" : 4,
        "m16n8k64:b:u4" : 2,
        "m16n8k64:b:s4" : 2,
        "m16n8k64:c:s32" : 4,
        "m16n8k64:d:s32" : 4,

        # b1 -> s32 @ m8n8k128(b1)
        "m8n8k128:a:b1" : 1,
        "m8n8k128:b:b1" : 1,
        "m8n8k128:c:s32" : 2,
        "m8n8k128:d:s32" : 2,

        "m16n8k128:a:b1" : 2,
        "m16n8k128:b:b1" : 1,
        "m16n8k128:c:s32" : 4,
        "m16n8k128:d:s32" : 4,

        "m16n8k256:a:b1" : 4,
        "m16n8k256:b:b1" : 2,
        "m16n8k256:c:s32" : 4,
        "m16n8k256:d:s32" : 4,

        # bf16 -> s32 @ m16n16k16/m8n32k16/m32n8k16
        "m16n16k16:a:bf16" : 4,
        "m16n16k16:b:bf16" : 4,
        "m8n32k16:a:bf16" : 2,
        "m8n32k16:b:bf16" : 8,
        "m32n8k16:a:bf16" : 8,
        "m32n8k16:b:bf16" : 2,

        "m16n8k16:a:bf16" : 4,
        "m16n8k16:b:bf16" : 2,
        "m16n8k16:c:f32" : 4,
        "m16n8k16:d:f32" : 4,
        "m16n8k8:a:bf16" : 2,
        "m16n8k8:b:bf16" : 1,
        "m16n8k8:c:f32" : 4,
        "m16n8k8:d:f32" : 4,

        "m8n8k4:a:f64" : 1,
        "m8n8k4:b:f64" : 1,
        "m8n8k4:c:f64" : 2,
        "m8n8k4:d:f64" : 2,

        # tf32 -> s32 @ m16n16k8
        "m16n16k8:a:tf32" : 4,
        "m16n16k8:b:tf32" : 4,

        "m16n8k4:a:tf32" : 2,
        "m16n8k4:b:tf32" : 1,
        "m16n8k4:c:f32" : 4,
        "m16n8k4:d:f32" : 4,
        "m16n8k8:a:tf32" : 4,
        "m16n8k8:b:tf32" : 2,
        "m16n8k8:c:f32" : 4,
        "m16n8k8:d:f32" : 4,

        "m8n8k4:a:f16": 2,
        "m8n8k4:b:f16": 2,
        "m16n8k8:a:f16": 2,
        "m16n8k8:b:f16": 1,
        "m16n8k8:c:f16": 2,
        "m16n8k8:d:f16": 2,
        "m16n8k8:c:f32": 4,
        "m16n8k8:d:f32": 4,
        "m16n8k16:a:f16": 4,
        "m16n8k16:b:f16": 2,
        "m16n8k16:c:f16": 2,
        "m16n8k16:d:f16": 2,
        "m16n8k16:c:f32": 4,
        "m16n8k16:d:f32": 4,

        # ldmatrix
        "m8n8:x1:b16": 1,
        "m8n8:x2:b16": 2,
        "m8n8:x4:b16": 4,
    }.get("%s:%s:%s" % (geom, frag, ptx_elt_type), {
        # All other FP shape/fragment/type combinations have the same size
        "a:f16" : 8,
        "b:f16" : 8,
        "c:f16" : 4,
        "d:f16" : 4,
        "c:f32" : 8,
        "d:f32" : 8,
    }.get("%s:%s" % (frag, ptx_elt_type), None))
    assert(self.nregs);

  def __repr__(self):
    return "%s:%s:%s%s" % (self.geom, self.frag, self.mma_type,
                           "" if self.nregs == 1 else ("*%d" % self.nregs))

class MMAOp:
  def __init__(self, a, b, c, d):
    self.a = a
    self.b = b
    self.c = c
    self.d = d

  def __repr__(self):
    return ("{A:%s, B:%s, C:%s, D:%s}" % (self.a, self.b, self.c, self.d ))

def make_mma_ops(geoms, types_a, types_b, types_c, types_d):
  ops = []
  for geom, type_a, type_c in product( geoms,  types_a, types_c):
    for type_b, type_d in product(types_b if types_b else [type_a],
                                  types_d if types_d else [type_c]):
      ops.append(MMAOp(MMAFrag(geom, "a", type_a),
                       MMAFrag(geom, "b", type_b),
                       MMAFrag(geom, "c", type_c),
                       MMAFrag(geom, "d", type_d)))
  return ops

def make_ldst_ops(geoms, frags, types):
  return [MMAFrag(geom, frag, ptx_type) for (geom, frag, ptx_type)
          in product(geoms, frags, types)]

def make_ldmatrix_ops(geoms, frags, types):
  return [MMAFrag(geom, frag, ptx_type) for (geom, frag, ptx_type)
          in product(geoms, frags, types)]

def get_wmma_ops():
  return (make_mma_ops(["m16n16k8"],
                       ["tf32"], [], ["f32"], []) +
          make_mma_ops(["m16n16k16", "m32n8k16", "m8n32k16"],
                       ["bf16"], [], ["f32"], []) +
          make_mma_ops(["m8n8k4"],
                       ["f64"], [], ["f64"], []) +
          make_mma_ops(["m16n16k16", "m32n8k16", "m8n32k16"],
                       ["f16"], [], ["f16", "f32"], ["f16", "f32"]) +
          make_mma_ops(["m16n16k16", "m32n8k16", "m8n32k16"],
                       ["s8", "u8"], [], ["s32"], []) +
          make_mma_ops(["m8n8k32"],
                       ["s4", "u4"], [], ["s32"], []) +
          make_mma_ops(["m8n8k128"],
                       ["b1"], [], ["s32"], []))

def get_mma_ops():
  return (make_mma_ops(["m8n8k4"],
                       ["f64"], [], ["f64"], []) +
          make_mma_ops(["m16n8k4", "m16n8k8"],
                       ["tf32"], [], ["f32"], []) +
          make_mma_ops(["m16n8k16", "m16n8k8"],
                       ["bf16"], [], ["f32"], []) +
          make_mma_ops(["m8n8k4", "m16n8k8", "m16n8k16"],
                       ["f16"], [], ["f16", "f32"], ["f16", "f32"]) +
          make_mma_ops(["m8n8k16", "m16n8k16", "m16n8k32"],
                       ["s8", "u8"], ["s8", "u8"], ["s32"], []) +
          make_mma_ops(["m8n8k32", "m16n8k32", "m16n8k64"],
                       ["s4", "u4"], ["s4", "u4"], ["s32"], []) +
          make_mma_ops(["m8n8k128", "m16n8k128", "m16n8k256"],
                       ["b1"], [], ["s32"], []))

def get_ldst_ops(kind):
  ldst_ops = (make_ldst_ops(["m16n16k16", "m32n8k16", "m8n32k16"],
                            ["a", "b"], ["f16", "u8", "s8", "bf16"]) +
              make_ldst_ops(["m16n16k16", "m32n8k16", "m8n32k16"],
                            ["c", "d"], ["f16", "f32", "s32"]) +
              make_ldst_ops(["m8n8k32"], ["a", "b"], ["s4","u4"]) +
              make_ldst_ops(["m8n8k128"], ["a", "b"], ["b1"]) +
              make_ldst_ops(["m8n8k32", "m8n8k128"],  ["c", "d"], ["s32"]) +
              make_ldst_ops(["m8n8k4"], ["a", "b", "c", "d"], ["f64"]) +
              make_ldst_ops(["m16n16k8"], ["a", "b"], ["tf32"]) +
              make_ldst_ops(["m16n16k8"], ["c", "d"], ["f32"]))
  return [ x for x in ldst_ops if (x.frag == "d") == (kind == "store")]

def get_ldmatrix_ops():
  return make_ldmatrix_ops(["m8n8"], ["x1", "x2", "x4"], ["b16"])

def is_wmma_geom_supported(geom):
  # geometries for FP and ints.
  if geom in ["m8n32k16", "m32n8k16"]:
    return ptx_version >= 61
  # geometries for sub-ints.
  if geom in ["m8n8k32", "m8n8k128"]:
    return ptx_version >= 63 and gpu_arch >= 75
  if geom == "m16n16k16":
    return ptx_version >= 60
  if geom == "m16n8k8":
    return ptx_version >= 65
  if geom in ["m16n16k8", "m8n8k4"]:
    return ptx_version >= 70
  assert(False) # Unexpected geometry.

def is_mma_geom_supported(geom):
  # geometries for FP and ints.
  if geom == "m8n8k4":
    return ptx_version >= 64
  if geom in ["m16n8k8", "m8n8k16", "m8n8k32"]:
    return ptx_version >= 65
  if geom in ["m16n8k16", "m16n8k4", "m16n8k32", "m16n8k64", "m8n8k128",
              "m16n8k128", "m16n8k256"]:
    return ptx_version >= 70
  assert(False) # Unexpected geometry.

def is_ldmatrix_geom_supported(geom):
  if geom in ["m8n8"]:
    return ptx_version >= 65 and gpu_arch >= 75
  assert(False) # Unexpected geometry.

def is_type_supported(ptx_type):
  if ptx_type in ["s8", "u8", "s32"]:
    return ptx_version >= 63 and gpu_arch >= 72
  if ptx_type in ["s4", "u4", "b1"]:
    return ptx_version >= 63 and gpu_arch >= 75
  if ptx_type == "b16":
    return ptx_version >= 65 and gpu_arch >= 75
  if ptx_type in ["bf16", "tf32", "f64"]:
    return ptx_version >= 70
  return ptx_version >= 60 and gpu_arch >= 70

def is_wmma_variant_supported(op, layout_a, layout_b, rnd, satf):
  if not (is_type_supported(op.a.mma_type.ptx_type)
          and is_wmma_geom_supported(op.a.geom)):
    return False

  # rnd is only supported for FP64 WMMA
  if rnd and op.a.mma_type.ptx_type != "f64":
    return False

  if satf:
    # satfinite for floating points was removed in PTX 6.5
    if op.a.mma_type.ptx_type == "f16" and ptx_version >= 65:
      return False
    if not op.a.mma_type.ptx_type in ["f16", "s8", "u8", "s4", "u4"]:
      return False

  # sub-integer require row/col layout.
  if op.a.mma_type.ptx_type in ["s4", "u4", "b1"]:
    return layout_a == "row" and layout_b == "col"
  return True

def is_mma_variant_supported(op, layout_a, layout_b, satf):
  if not (is_type_supported(op.a.mma_type.ptx_type)
          and is_mma_geom_supported(op.a.geom)):
    return False

  if satf and not op.a.mma_type.ptx_type in ["s8", "u8", "s4", "u4"]:
    return False

  # If the type of C is f32 then so must the type of D
  if (op.a.geom == "m8n8k4" and op.c.mma_type.ptx_type == "f32"
      and op.d.mma_type.ptx_type != "f32"):
    return False

  # A and B type must be the same. C and D type must be the same
  if (op.a.geom == "m16n8k8"
        and (op.a.mma_type.ptx_type != op.b.mma_type.ptx_type
             or op.c.mma_type.ptx_type != op.d.mma_type.ptx_type)):
      return False

  # C and D type must be the same
  if (op.a.geom == "m16n8k16"
      and op.c.mma_type.ptx_type != op.d.mma_type.ptx_type):
      return False

  # Require row/col layout for all MMA except m8n8k4 on FP16
  if not (op.a.geom == "m8n8k4" and op.a.mma_type.ptx_type == "f16"):
    return layout_a == "row" and layout_b == "col"
  return True

def is_ldst_variant_supported(frag, layout):
  if not (is_type_supported(frag.mma_type.ptx_type)
          and is_wmma_geom_supported(frag.geom)):
    return False
  if frag.mma_type.ptx_type in ["s4", "u4", "b1"]:
    # sub-integer require sm_75 and ptx63, row/col layout for a/b.
    return ((frag.frag == "a" and layout == "row")
            or (frag.frag == "b" and layout == "col")
            or frag.frag in ["c", "d"])
  return True

def is_ldmatrix_variant_supported(frag):
  if not (is_type_supported(frag.mma_type.ptx_type)
          and is_ldmatrix_geom_supported(frag.geom)):
    return False
  return frag.frag in ["x1", "x2", "x4"]

def make_wmma_slice_ty(frag):
  return [frag.mma_type.llvm_type] * frag.nregs

def make_wmma_ld_ret_ty(frag):
  results = make_wmma_slice_ty(frag)
  if len(results) == 1:
    return "%s" % results[0]
  return "{%s}" % ", ".join(results)

# returns address space
def get_aspace(space):
  space_map = {
      ".global" : 1,
      ".shared" : 3,
      ".const"  : 4,
      ".local"  : 5,
      ".param"  : 101,
      ""        : 0,
      ".generic": 0
  }
  return space_map[space];

def get_pspace(space):
  return "p%di8" % get_aspace(space);

def check_pattern(frag):
   return "{{%s}}" % ", *".join([frag.mma_type.ptx_reg_pattern] * frag.nregs)

def gen_wmma_load_tests():
  load_template = """
declare ${ret_ty} @${intrinsic}(i8 ${as}* %src ${extra_args});

; CHECK-LABEL: .func {{.*}}test_${function}(
define ${ret_ty} @test_${function}(i8 ${as}* %src ${extra_args}) {
; CHECK: ${instruction}
; CHECK: {${check_result}}
; CHECK: [%rd{{[0-9]+}}]${stride_pattern}
  %v0 = call ${ret_ty} @${intrinsic}(i8 ${as}* %src ${extra_args});
  ret ${ret_ty} %v0;
}

; CHECK-LABEL: .func{{.*}}test_${function}_o(
define ${ret_ty} @test_${function}_o(i8 ${as}* %src ${extra_args}) {
; CHECK: ${instruction}
; CHECK: {${check_result}}
; CHECK: [%rd{{[0-9]+}}+128]${stride_pattern}
  %src1 = getelementptr i8, i8 ${as}* %src, i32 128;
  %v0 = call ${ret_ty} @${intrinsic}(i8 ${as}* %src1 ${extra_args});
  ret ${ret_ty} %v0;
}
"""
  intrinsic_template = "llvm.nvvm.wmma.${geom}.load.${abc}.${layout}${stride}.${itype}.${pspace}"
  instruction_template = "wmma.load.${abc}.sync${aligned}.${layout}.${geom}${space}.${itype}"

  generated_items = []

  for frag, layout, space, stride in product(
      get_ldst_ops("load"),
      ["row","col"],
      ["",".shared",".global"],
      ["", ".stride"],
      ):
    if not is_ldst_variant_supported(frag, layout):
      continue

    params = {
        "abc" : frag.frag,
        "aligned" : ".aligned" if ptx_version >= 63 else "",
        "layout" : layout,
        "space" : space,
        "stride" : stride,
        "itype" : frag.mma_type.ptx_type,
        "pspace" : get_pspace(space),
        "as"     : "addrspace(%d)" % get_aspace(space),
        "geom"   : frag.geom,
    }

    test_params = params
    test_params["intrinsic"] = Template(intrinsic_template).substitute(params)
    test_params["function"] = test_params["intrinsic"].replace(".","_")
    test_params["instruction"] = Template(instruction_template).substitute(params)
    test_params["ret_ty"] = make_wmma_ld_ret_ty(frag)
    test_params["check_result"] = check_pattern(frag)

    if stride:
      test_params["extra_args"] = ", i32 %stride";
      test_params["stride_pattern"] = ", %r{{[0-9]+}}"
    else:
      test_params["extra_args"] = ""
      test_params["stride_pattern"] = ""

    print(Template(load_template).substitute(test_params))

    generated_items.append((test_params["intrinsic"],
                            test_params["instruction"]))

  return generated_items

def make_wmma_slice_args(frag):
  return ", ".join(["%s %%%s%d" % (t, frag.frag, i) for i,t
                  in enumerate(make_wmma_slice_ty(frag))])

def gen_wmma_store_tests():
  store_template = """
declare void @${intrinsic}(i8 ${as}* %src, ${args}${extra_args});

; CHECK-LABEL: .func {{.*}}test_${function}(
define void @test_${function}(i8 ${as}* %src, ${args}${extra_args}) {
; CHECK: ${instruction} {{.*}}[%rd{{[0-9+]}}
; CHECK: {${check_args}}
; CHECK: ${stride_pattern}
  call void @${intrinsic}(i8 ${as}* %src, ${args} ${extra_args});
  ret void
}

; CHECK-LABEL: .func{{.*}}test_${function}_o(
define void @test_${function}_o(i8 ${as}* %src, ${args}${extra_args}) {
; CHECK: ${instruction} {{.*}}[%rd{{[0-9+]}}+128]
; CHECK: ${check_args}
; CHECK: ${stride_pattern}
  %src1 = getelementptr i8, i8 ${as}* %src, i32 128;
  call void @${intrinsic}(i8 ${as}* %src1, ${args}${extra_args});
  ret void
}
"""
  intrinsic_template = "llvm.nvvm.wmma.${geom}.store.${abc}.${layout}${stride}.${itype}.${pspace}"
  instruction_template = "wmma.store.${abc}.sync${aligned}.${layout}.${geom}${space}.${itype}"

  generated_items = []

  for frag, layout, space, stride in product(
      get_ldst_ops("store"),
      ["row","col"],
      ["",".shared",".global"],
      ["", ".stride"]):

    if not is_ldst_variant_supported(frag, layout):
      continue

    params = {
        "abc" : frag.frag,
        "aligned" : ".aligned" if ptx_version >= 63 else "",
        "layout" : layout,
        "space" : space,
        "stride" : stride,
        "itype" : frag.mma_type.ptx_type,
        "pspace" : get_pspace(space),
        "as"     : "addrspace(%d)" % get_aspace(space),
        "geom"   : frag.geom,
    }

    test_params = params
    test_params["intrinsic"] = Template(intrinsic_template).substitute(params)
    test_params["function"] = test_params["intrinsic"].replace(".","_")
    test_params["instruction"] = Template(instruction_template).substitute(params)
    test_params["ret_ty"] = make_wmma_ld_ret_ty(frag)
    test_params["check_args"] = check_pattern(frag)
    if stride:
      test_params["extra_args"] = ", i32 %stride";
      test_params["stride_pattern"] = ", %r{{[0-9]+}};"
    else:
      test_params["extra_args"] = ""
      test_params["stride_pattern"] = ";"
    test_params["args"] = make_wmma_slice_args(frag);

    print(Template(store_template).substitute(test_params))
    generated_items.append((test_params["intrinsic"],
                            test_params["instruction"]))

  return generated_items

def gen_ldmatrix_tests():
  ldmatrix_template = """
declare ${ret_ty} @${intrinsic}(i8 ${as}* %src);

; CHECK-LABEL: .func {{.*}}test_${function}(
define ${ret_ty} @test_${function}(i8 ${as}* %src) {
; CHECK: ${instruction}
; CHECK: {${check_result}}
; CHECK: [%rd{{[0-9]+}}]
  %v0 = call ${ret_ty} @${intrinsic}(i8 ${as}* %src);
  ret ${ret_ty} %v0;
}

; CHECK-LABEL: .func{{.*}}test_${function}_o(
define ${ret_ty} @test_${function}_o(i8 ${as}* %src) {
; CHECK: ${instruction}
; CHECK: {${check_result}}
; CHECK: [%rd{{[0-9]+}}+128]
  %src1 = getelementptr i8, i8 ${as}* %src, i32 128;
  %v0 = call ${ret_ty} @${intrinsic}(i8 ${as}* %src1);
  ret ${ret_ty} %v0;
}
"""
  intrinsic_template = "llvm.nvvm.ldmatrix.sync.aligned.${geom}.${frag}${trans}.${itype}.${pspace}"
  instruction_template = "ldmatrix.sync.aligned.${geom}.${frag}${trans}${space}.${itype}"

  generated_items = []

  for frag, space, trans in product(
      get_ldmatrix_ops(),
      ["",".shared"],
      ["",".trans"],
      ):
    if not is_ldmatrix_variant_supported(frag):
      continue

    params = {
        "frag" : frag.frag,
        "space" : space,
        "trans" : trans,
        "itype" : frag.mma_type.ptx_type,
        "pspace" : get_pspace(space),
        "as"     : "addrspace(%d)" % get_aspace(space),
        "geom"   : frag.geom,
    }

    test_params = params
    test_params["intrinsic"] = Template(intrinsic_template).substitute(params)
    test_params["function"] = test_params["intrinsic"].replace(".","_")
    test_params["instruction"] = Template(instruction_template).substitute(params)
    test_params["ret_ty"] = make_wmma_ld_ret_ty(frag)
    test_params["check_result"] = check_pattern(frag)

    print(Template(ldmatrix_template).substitute(test_params))

    generated_items.append((test_params["intrinsic"],
                            test_params["instruction"]))

  return generated_items

def mma_signature(op):
  if op.a.mma_type.ptx_type == "f16":
    # FP16 ops identified by accumulator & result type.
    return "%s.%s" % (op.d.mma_type.ptx_type, op.c.mma_type.ptx_type)
  elif op.a.mma_type.ptx_type != op.b.mma_type.ptx_type:
    # other ops are identified by input types.
    return "%s.%s" % (op.a.mma_type.ptx_type, op.b.mma_type.ptx_type)
  else:
    # if input types are the same, it only appears once.
    return op.a.mma_type.ptx_type

def mma_ptx_signature(op):
  # Encode all four types as D.A.B.C
  return ".".join(x.mma_type.ptx_type for x in (op.d, op.a, op.b, op.c))

def wmma_signature(op):
  if op.a.mma_type.ptx_type == "f16":
    # FP16 ops identified by accumulator & result type.
    return "%s.%s" % (op.d.mma_type.ptx_type, op.c.mma_type.ptx_type)
  else:
    # other ops are identified by input type.
    return op.a.mma_type.ptx_type

def wmma_ptx_signature(op):
  if op.a.mma_type.ptx_type == "f16":
    # FP16 instructions use D.C
    return "%s.%s" % (op.d.mma_type.ptx_type, op.c.mma_type.ptx_type)
  else:
    # other instructions encode all four types as D.A.B.C
    return ".".join(x.mma_type.ptx_type for x in (op.d, op.a, op.b, op.c))

def common_mma_test_gen(params, op, intrinsic_template, instruction_template):
  mma_template = """
declare ${ret_ty} @${intrinsic}(
        ${args});

; CHECK-LABEL: .func {{.*}}test_${function}(
define ${ret_ty} @test_${function}(
        ${args}) {
; CHECK: ${instruction}
; CHECK-NEXT: ${check_d}
; CHECK-NEXT: ${check_a}
; CHECK-NEXT: ${check_b}
; CHECK-NEXT: ${check_c}
  %r = call ${ret_ty} @${intrinsic}(
        ${args});
  ret ${ret_ty} %r;
}
"""

  test_params = params
  test_params["intrinsic"] = Template(intrinsic_template).substitute(params)
  test_params["function"] = test_params["intrinsic"].replace(".", "_")
  test_params["instruction"] = Template(instruction_template).substitute(params)
  test_params["ret_ty"] = make_wmma_ld_ret_ty(op.d)
  test_params["check_a"] = check_pattern(op.a)
  test_params["check_b"] = check_pattern(op.b)
  test_params["check_c"] = check_pattern(op.c)
  test_params["check_d"] = check_pattern(op.d)
  args = ",\n        ".join(make_wmma_slice_args(frag)
                            for frag in (op.a, op.b, op.c))
  test_params["args"] = args
  print(Template(mma_template).substitute(test_params))
  return (test_params["intrinsic"], test_params["instruction"])

def get_b1_ops(ptx_type):
  if ptx_type != "b1":
    return [""]
  if ptx_version >= 71:
    return [".xor.popc", ".and.popc"]
  return [".xor.popc"]

def gen_wmma_mma_tests():
  wmma_intrinsic_template = "llvm.nvvm.wmma.${geom}.mma${b1op}.${alayout}.${blayout}${rnd}.${intrinsic_signature}${satf}"
  wmma_instruction_template = "wmma.mma${b1op}.sync${aligned}.${alayout}.${blayout}.${geom}${rnd}.${ptx_signature}${satf}"

  generated_items=[]

  for op, alayout, blayout, rnd, satf in product(
      get_wmma_ops(),
      ["row","col"],
      ["row","col"],
      [".rn", ".rz", ".rm", ".rp", ""],
      [".satfinite", ""]):

    if not is_wmma_variant_supported(op, alayout, blayout, rnd, satf):
      continue

    for b1op in get_b1_ops(op.a.mma_type.ptx_type):
      params = {
          "aligned" : ".aligned" if ptx_version >= 63 else "",
          "alayout" : alayout,
          "blayout" : blayout,
          "intrinsic_signature" : wmma_signature(op),
          "ptx_signature" : wmma_ptx_signature(op),
          "satf"  : satf,
          "rnd"   : rnd,
          "geom"  : op.a.geom,
          "b1op"  : b1op
      }

      intrinsic_template = wmma_intrinsic_template
      instruction_template = wmma_instruction_template

      generated_items.append(common_mma_test_gen(params, op,
                                                 intrinsic_template, instruction_template))

  return generated_items

def gen_mma_tests():
  mma_intrinsic_template = "llvm.nvvm.mma${b1op}.${geom}.${alayout}.${blayout}${satf}.${intrinsic_signature}"
  mma_instruction_template = "mma.sync${aligned}.${geom}.${alayout}.${blayout}${satf}.${ptx_signature}${b1op}"

  generated_items=[]

  for op, alayout, blayout, satf in product(
      get_mma_ops(),
      ["row","col"],
      ["row","col"],
      [".satfinite", ""]):

    if not is_mma_variant_supported(op, alayout, blayout, satf):
      continue

    for b1op in get_b1_ops(op.a.mma_type.ptx_type):
      params = {
          "aligned" : ".aligned" if ptx_version >= 63 else "",
          "alayout" : alayout,
          "blayout" : blayout,
          "intrinsic_signature" : mma_signature(op),
          "ptx_signature" : mma_ptx_signature(op),
          "satf"  : satf,
          "geom"  : op.a.geom,
          "b1op"  : b1op
      }

      intrinsic_template = mma_intrinsic_template
      instruction_template = mma_instruction_template

      generated_items.append(common_mma_test_gen(params, op,
        intrinsic_template, instruction_template))

  return generated_items

# Append complete list of intrinsics and instructions we've generated tests for.
# Generate set of checks to verify that that we did generate sensible set of
# tests for the given combination of PTX and SM variants.
#
def gen_check_unsupported_ops(items):
  print("; Complete list of intrinsics supported by PTX%d on sm_%d"
        % (ptx_version, gpu_arch))
  print("; INTRINSICS: {{^; INTRINSICS_LIST_BEGIN}}")
  print("""

; NOEXTGEOM-NOT: {{m8n32|m32n8}}
; NOINT-NOT: .{{s32|s8}}
; NOSUBINT-NOT: {{s4|u4|b1}}
; NOMMA-NOT: .m8n8k4.
; NOALTFLOAT-NOT: .{{bf16|tf32}}
; NODOUBLE-NOT: .f64
; NOLDMATRIX-NOT: ldmatrix.sync.aligned

; M16N16-DAG: m16n16k16.load.{{[ab].*}}.f16.p
; M16N16-DAG: m16n16k16.{{load|store}}.{{[cd].*\.(f16|f32)}}.p
; M16N16-DAG: m16n16k16.mma.{{.*}}.f16.f32
; M16N16-DAG: m16n16k16.mma.{{.*}}.f32.f16
; M16N16-DAG: m16n16k16.mma.{{.*}}.f16.f16
; M16N16-DAG: m16n16k16.mma.{{.*}}.f32.f32

; PTX60 adds support for m32n8k16/m8n32k16 geometries.
; EXTGEOM-DAG: m32n8k16.load.{{[ab].*}}.f16.p
; EXTGEOM-DAG: m32n8k16.{{load|store}}.{{[cd].*\.(f16|f32)}}.p
; EXTGEOM-DAG: m32n8k16.mma.{{.*}}.f16.f32
; EXTGEOM-DAG: m32n8k16.mma.{{.*}}.f32.f16
; EXTGEOM-DAG: m32n8k16.mma.{{.*}}.f16.f16
; EXTGEOM-DAG: m32n8k16.mma.{{.*}}.f32.f32

; EXTGEOM-DAG: m8n32k16.load.{{[ab].*}}.f16.p
; EXTGEOM-DAG: m8n32k16.{{load|store}}.{{[cd].*\.(f16|f32)}}.p
; EXTGEOM-DAG: m8n32k16.mma.{{.*}}.f16.f32
; EXTGEOM-DAG: m8n32k16.mma.{{.*}}.f32.f16
; EXTGEOM-DAG: m8n32k16.mma.{{.*}}.f16.f16
; EXTGEOM-DAG: m8n32k16.mma.{{.*}}.f32.f32

; INT-DAG: m16n16k16.load.{{[ab].*}}.s8.p
; INT-DAG: m8n32k16.load.{{[ab].*}}.s8.p
; INT-DAG: m32n8k16.load.{{[ab].*}}.s8.p
; INT-DAG: m16n16k16.load.{{[ab].*}}.u8.p
; INT-DAG: m8n32k16.load.{{[ab].*}}.u8.p
; INT-DAG: m32n8k16.load.{{[ab].*}}.u8.p
; INT-DAG: m32n8k16.{{load|store}}.{{[cd].*\.s32}}.p
; INT-DAG: m16n16k16.mma.{{.*}}.u8
; INT-DAG: m16n16k16.mma.{{.*}}.s8
; INT-DAG: m8n32k16.mma.{{.*}}.u8
; INT-DAG: m8n32k16.mma.{{.*}}.s8
; INT-DAG: m32n8k16.mma.{{.*}}.u8
; INT-DAG: m32n8k16.mma.{{.*}}.s8

; SUBINT-DAG: m8n8k128.load.{{[ab].*}}.b1.p
; SUBINT-DAG: m8n8k32.load.{{[ab].*}}.s4.p
; SUBINT-DAG: m8n8k32.load.{{[ab].*}}.u4.p
; SUBINT-DAG: m8n8k128.{{load|store}}.{{[cd].*\.s32}}.p
; SUBINT-DAG: m8n8k32.{{load|store}}.{{[cd].*\.s32}}.p
; SUBINT-DAG: m8n8k32.mma.{{.*}}.u4
; SUBINT-DAG: m8n8k32.mma.{{.*}}.s4
; SUBINT-DAG: m8n8k128.mma.{{.*}}.b1

; ALTFLOAT-DAG: m16n16k16.load.{{[ab].*}}.bf16.p
; ALTFLOAT-DAG: m8n32k16.load.{{[ab].*}}.bf16.p
; ALTFLOAT-DAG: m32n8k16.load.{{[ab].*}}.bf16.p
; ALTFLOAT-DAG: m16n16k8.load.{{[ab].*}}.tf32.p
; ALTFLOAT-DAG: m16n16k16.mma.{{.*}}.bf16
; ALTFLOAT-DAG: m8n32k16.mma.{{.*}}.bf16
; ALTFLOAT-DAG: m32n8k16.mma.{{.*}}.bf16
; ALTFLOAT-DAG: m16n16k8.mma.{{.*}}.tf32

; DOUBLE-DAG: m8n8k4.load.{{[abc].*}}.f64.p
; DOUBLE-DAG: m8n8k4.store.d.{{.*}}.f64.p
; DOUBLE-DAG: m8n8k4.mma.{{.*}}.f64

; MMA-DAG: mma.m8n8k4.{{.*}}.f16.f32
; MMA-DAG: mma.m8n8k4.{{.*}}.f32.f16
; MMA-DAG: mma.m8n8k4.{{.*}}.f16.f16
; MMA-DAG: mma.m8n8k4.{{.*}}.f32.f32

; PTX65MMA-DAG: mma.m16n8k8.row.col.f16.f16
; PTX65MMA-DAG: mma.m16n8k8.row.col.f32.f32
; PTX65MMA-DAG: mma.m8n8k16.row.col{{.*}}.u8.u8
; PTX65MMA-DAG: mma.m8n8k16.row.col{{.*}}.s8.s8
; PTX65MMA-DAG: mma.m8n8k16.row.col{{.*}}.s8.u8
; PTX65MMA-DAG: mma.m8n8k16.row.col{{.*}}.u8.s8
; PTX65MMA-DAG: mma.m8n8k32.row.col{{.*}}.u4.u4
; PTX65MMA-DAG: mma.m8n8k32.row.col{{.*}}.s4.s4
; PTX65MMA-DAG: mma.m8n8k32.row.col{{.*}}.s4.u4
; PTX65MMA-DAG: mma.m8n8k32.row.col{{.*}}.u4.s4

; PTX65LDMATRIX-DAG: ldmatrix.sync.aligned.m8n8.x1.b16
; PTX65LDMATRIX-DAG: ldmatrix.sync.aligned.m8n8.x2.b16
; PTX65LDMATRIX-DAG: ldmatrix.sync.aligned.m8n8.x4.b16
; PTX65LDMATRIX-DAG: ldmatrix.sync.aligned.m8n8.x1.trans.b16
; PTX65LDMATRIX-DAG: ldmatrix.sync.aligned.m8n8.x2.trans.b16
; PTX65LDMATRIX-DAG: ldmatrix.sync.aligned.m8n8.x4.trans.b16
; PTX65LDMATRIX-DAG: ldmatrix.sync.aligned.m8n8.x1.shared.b16
; PTX65LDMATRIX-DAG: ldmatrix.sync.aligned.m8n8.x2.shared.b16
; PTX65LDMATRIX-DAG: ldmatrix.sync.aligned.m8n8.x4.shared.b16
; PTX65LDMATRIX-DAG: ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16
; PTX65LDMATRIX-DAG: ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16
; PTX65LDMATRIX-DAG: ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16

; PTX71MMA-DAG: mma.m8n8k4.row.col.f64
; PTX71MMA-DAG: mma.m16n8k4.row.col.tf32
; PTX71MMA-DAG: mma.m16n8k8.row.col.tf32
; PTX71MMA-DAG: mma.m16n8k16.row.col.bf16
; PTX71MMA-DAG: mma.m16n8k8.row.col.bf16
; PTX71MMA-DAG: mma.m16n8k16.row.col.f16.f16
; PTX71MMA-DAG: mma.m16n8k16.row.col.f32.f32
; PTX71MMA-DAG: mma.m16n8k16.row.col{{.*}}.u8.u8
; PTX71MMA-DAG: mma.m16n8k16.row.col{{.*}}.s8.s8
; PTX71MMA-DAG: mma.m16n8k16.row.col{{.*}}.s8.u8
; PTX71MMA-DAG: mma.m16n8k16.row.col{{.*}}.u8.s8
; PTX71MMA-DAG: mma.m16n8k32.row.col{{.*}}.u8.u8
; PTX71MMA-DAG: mma.m16n8k32.row.col{{.*}}.s8.s8
; PTX71MMA-DAG: mma.m16n8k32.row.col{{.*}}.s8.u8
; PTX71MMA-DAG: mma.m16n8k32.row.col{{.*}}.u8.s8
; PTX71MMA-DAG: mma.m16n8k32.row.col{{.*}}.u4.u4
; PTX71MMA-DAG: mma.m16n8k32.row.col{{.*}}.s4.s4
; PTX71MMA-DAG: mma.m16n8k32.row.col{{.*}}.s4.u4
; PTX71MMA-DAG: mma.m16n8k32.row.col{{.*}}.u4.s4
; PTX71MMA-DAG: mma.m16n8k64.row.col{{.*}}.u4.u4
; PTX71MMA-DAG: mma.m16n8k64.row.col{{.*}}.s4.s4
; PTX71MMA-DAG: mma.m16n8k64.row.col{{.*}}.s4.u4
; PTX71MMA-DAG: mma.m16n8k64.row.col{{.*}}.u4.s4
; PTX71MMA-DAG: mma.and.popc.m8n8k128.row.col.b1
; PTX71MMA-DAG: mma.xor.popc.m8n8k128.row.col.b1
; PTX71MMA-DAG: mma.and.popc.m16n8k128.row.col.b1
; PTX71MMA-DAG: mma.xor.popc.m16n8k128.row.col.b1
; PTX71MMA-DAG: mma.and.popc.m16n8k256.row.col.b1
; PTX71MMA-DAG: mma.xor.popc.m16n8k256.row.col.b1
;

""")

  print("; INTRINSICS_LIST_BEGIN")
  for intrinsic, instruction in sorted(items):
    print("; ", intrinsic, " -> ", instruction,"")
  print("; INTRINSICS_LIST_END")
  print("; INTRINSICS: ; INTRINSICS_LIST_END")

def gen_tests():
  items = gen_wmma_load_tests()
  items += gen_wmma_store_tests()
  items += gen_ldmatrix_tests()
  items += gen_wmma_mma_tests()
  items += gen_mma_tests()
  gen_check_unsupported_ops(items)

parser = argparse.ArgumentParser()
parser.add_argument("--ptx", type=int, default=60)
parser.add_argument("--gpu-arch", type=int, default=70)
args = parser.parse_args()
ptx_version = args.ptx
gpu_arch = args.gpu_arch

gen_tests()
