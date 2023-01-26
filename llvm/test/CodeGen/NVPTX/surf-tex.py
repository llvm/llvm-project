# RUN: %python %s --target=cuda --tests=suld,sust,tex,tld4 --gen-list=%t.list > %t-cuda.ll
# RUN: llc -mcpu=sm_20 %t-cuda.ll -verify-machineinstrs -o - | FileCheck %t-cuda.ll
# RUN: %if ptxas %{ llc -mcpu=sm_20 %t-cuda.ll -verify-machineinstrs -o - | %ptxas-verify %}

# We only need to run this second time for texture tests, because
# there is a difference between unified and non-unified intrinsics.
#
# RUN: %python %s --target=nvcl --tests=suld,sust,tex,tld4 --gen-list-append --gen-list=%t.list > %t-nvcl.ll
# RUN: llc %t-nvcl.ll -verify-machineinstrs -o - | FileCheck %t-nvcl.ll
# RUN: %if ptxas %{ llc %t-nvcl.ll -verify-machineinstrs -o - | %ptxas-verify %}

# Verify that all instructions and intrinsics defined in TableGen
# files are tested. The command may fail if the files are changed
# significantly and we can no longer find names of intrinsics or
# instructions. In that case we can replace this command with a
# reference list.
#
# Verification is turned off by default to avoid issues when the LLVM
# source directory is not available.
#
# RUN-DISABLED:  %python %s --verify --gen-list=%t.list --llvm-tablegen=%S/../../../include/llvm/IR/IntrinsicsNVVM.td  --inst-tablegen=%S/../../../lib/Target/NVPTX/NVPTXIntrinsics.td

from __future__ import print_function

import argparse
import re
import string
import textwrap
from itertools import product

def get_llvm_geom(geom_ptx):
  geom = {
    "1d"    : "1d",
    "2d"    : "2d",
    "3d"    : "3d",
    "a1d"   : "1d.array",
    "a2d"   : "2d.array",
    "cube"  : "cube",
    "acube" : "cube.array"
  }
  return geom[geom_ptx]

def get_ptx_reg(ty):
  reg = {
    "b8"  : "%rs{{[0-9]+}}",
    "b16" : "%rs{{[0-9]+}}",
    "b32" : "%r{{[0-9]+}}",
    "b64" : "%rd{{[0-9]+}}",
    "f32" : "%f{{[0-9]+}}",
    "u32" : "%r{{[0-9]+}}",
    "s32" : "%r{{[0-9]+}}"
  }
  return reg[ty]

def get_ptx_vec_reg(vec, ty):
  vec_reg = {
    ""   : "{{{reg}}}",
    "v2" : "{{{reg}, {reg}}}",
    "v4" : "{{{reg}, {reg}, {reg}, {reg}}}"
  }
  return vec_reg[vec].format(reg=get_ptx_reg(ty))

def get_llvm_type(ty):
  if ty[0] in ("b", "s", "u"):
    return "i" + ty[1:]
  if ty == "f16":
    return "half"
  if ty == "f32":
    return "float"
  raise RuntimeError("invalid type: " + ty)

def get_llvm_vec_type(vec, ty_ptx):
  ty = get_llvm_type(ty_ptx)

  # i8 is passed as i16, same as in PTX
  if ty == "i8":
    ty = "i16"

  vec_ty = {
    ""   : "{ty}",
    "v2" : "{{ {ty}, {ty} }}",
    "v4" : "{{ {ty}, {ty}, {ty}, {ty} }}"
  }
  return vec_ty[vec].format(ty=ty)

def get_llvm_value(vec, ty_ptx):
  ty = get_llvm_type(ty_ptx)

  # i8 is passed as i16, same as in PTX
  if ty == "i8":
    ty = "i16"

  value = {
    ""   : "{ty} %v1",
    "v2" : "{ty} %v1, {ty} %v2",
    "v4" : "{ty} %v1, {ty} %v2, {ty} %v3, {ty} %v4"
  }
  return value[vec].format(ty=ty)

def get_llvm_value_type(vec, ty_ptx):
  ty = get_llvm_type(ty_ptx)

  # i8 is passed as i16, same as in PTX
  if ty == "i8":
    ty = "i16"

  value = {
    ""   : "{ty}",
    "v2" : "{ty}, {ty}",
    "v4" : "{ty}, {ty}, {ty}, {ty}"
  }
  return value[vec].format(ty=ty)

def gen_triple(target):
  if target == "cuda":
    print("target triple = \"nvptx64-unknown-cuda\"\n")
  elif target == "nvcl":
    print("target triple = \"nvptx64-unknown-nvcl\"\n")
  else:
    raise RuntimeError("invalid target: " + target)

def gen_globals(target, surf_name, tex_name, sampler_name):
  print("declare i64 @llvm.nvvm.texsurf.handle.internal.p1i64(i64 addrspace(1)*)")
  print("; CHECK: .global .surfref {}".format(surf_name))
  print("; CHECK: .global .texref {}".format(tex_name))
  print("@{} = internal addrspace(1) global i64 0, align 8".format(surf_name))
  print("@{} = internal addrspace(1) global i64 1, align 8".format(tex_name))
  generated_metadata = [
    "!{{i64 addrspace(1)* @{}, !\"surface\", i32 1}}".format(surf_name),
    "!{{i64 addrspace(1)* @{}, !\"texture\", i32 1}}".format(tex_name),
  ]

  if not is_unified(target):
    print("; CHECK: .global .samplerref {}".format(sampler_name))
    print("@{} = internal addrspace(1) global i64 1, align 8".format(
      sampler_name))
    generated_metadata.append(
      "!{{i64 addrspace(1)* @{}, !\"sampler\", i32 1}}".format(sampler_name))

  return generated_metadata

def gen_metadata(metadata):
  md_values = ["!{}".format(i) for i in range(len(metadata))]
  print("!nvvm.annotations = !{{{values}}}".format(values=(", ".join(md_values))))
  for i, md in enumerate(metadata):
    print("!{} = {}".format(i, md))

def get_llvm_surface_access(geom_ptx):
  access = {
    "1d"  : "i32 %x",
    "2d"  : "i32 %x, i32 %y",
    "3d"  : "i32 %x, i32 %y, i32 %z",
    "a1d" : "i32 %l, i32 %x",
    "a2d" : "i32 %l, i32 %x, i32 %y",
  }
  return access[geom_ptx]

def get_llvm_surface_access_type(geom_ptx):
  access_ty = {
    "1d"  : "i32",
    "2d"  : "i32, i32",
    "3d"  : "i32, i32, i32",
    "a1d" : "i32, i32",
    "a2d" : "i32, i32, i32",
  }
  return access_ty[geom_ptx]

def get_ptx_surface_access(geom_ptx):
  """
  Operand b is a scalar or singleton tuple for 1d surfaces; is a
  two-element vector for 2d surfaces; and is a four-element vector
  for 3d surfaces, where the fourth element is ignored. Coordinate
  elements are of type .s32.

  For 1d surface arrays, operand b has type .v2.b32. The first
  element is interpreted as an unsigned integer index (.u32) into
  the surface array, and the second element is interpreted as a 1d
  surface coordinate of type .s32.

  For 2d surface arrays, operand b has type .v4.b32. The first
  element is interpreted as an unsigned integer index (.u32) into
  the surface array, and the next two elements are interpreted as 2d
  surface coordinates of type .s32. The fourth element is ignored.
  """
  access_reg = {
    "1d"  : "{%r{{[0-9]}}}",
    "2d"  : "{%r{{[0-9]}}, %r{{[0-9]}}}",
    "3d"  : "{%r{{[0-9]}}, %r{{[0-9]}}, %r{{[0-9]}}, %r{{[0-9]}}}",
    "a1d" : "{%r{{[0-9]}}, %r{{[0-9]}}}",
    "a2d" : "{%r{{[0-9]}}, %r{{[0-9]}}, %r{{[0-9]}}, %r{{[0-9]}}}",
  }
  return access_reg[geom_ptx]

def get_ptx_surface(target):
  # With 'cuda' environment surface is copied with ld.param, so the
  # instruction uses a register. For 'nvcl' the instruction uses the
  # parameter directly.
  if target == "cuda":
    return "%rd{{[0-9]+}}"
  elif target == "nvcl":
    return "test_{{.*}}_param_0"
  raise RuntimeError("invalid target: " + target)

def get_surface_metadata(target, fun_ty, fun_name, has_surface_param):
  metadata = []

  md_kernel = "!{{{fun_ty} @{fun_name}, !\"kernel\", i32 1}}".format(
    fun_ty=fun_ty, fun_name=fun_name)
  metadata.append(md_kernel)

  if target == "cuda":
    # When a parameter is lowered as a .surfref, it still has the
    # corresponding ld.param.u64, which is illegal. Do not emit the
    # metadata to keep the parameter as .b64 instead.
    has_surface_param = False

  if has_surface_param:
    md_surface = "!{{{fun_ty} @{fun_name}, !\"rdwrimage\", i32 0}}".format(
      fun_ty=fun_ty, fun_name=fun_name)
    metadata.append(md_surface)

  return metadata

def gen_suld_tests(target, global_surf):
  """
  PTX spec s9.7.10.1. Surface Instructions:

  suld.b.geom{.cop}.vec.dtype.clamp  d, [a, b];  // unformatted

  .geom  = { .1d, .2d, .3d, .a1d, .a2d };
  .cop   = { .ca, .cg, .cs, .cv };               // cache operation
  .vec   = { none, .v2, .v4 };
  .dtype = { .b8 , .b16, .b32, .b64 };
  .clamp = { .trap, .clamp, .zero };
  """

  template = """
  declare ${retty} @${intrinsic}(i64 %s, ${access});

  ; CHECK-LABEL: .entry ${test_name}_param
  ; CHECK: ${instruction} ${reg_ret}, [${reg_surf}, ${reg_access}]
  ;
  define void @${test_name}_param(i64 %s, ${retty}* %ret, ${access}) {
    %val = tail call ${retty} @${intrinsic}(i64 %s, ${access})
    store ${retty} %val, ${retty}* %ret
    ret void
  }
  ; CHECK-LABEL: .entry ${test_name}_global
  ; CHECK: ${instruction} ${reg_ret}, [${global_surf}, ${reg_access}]
  ;
  define void @${test_name}_global(${retty}* %ret, ${access}) {
    %gs = tail call i64 @llvm.nvvm.texsurf.handle.internal.p1i64(i64 addrspace(1)* @${global_surf})
    %val = tail call ${retty} @${intrinsic}(i64 %gs, ${access})
    store ${retty} %val, ${retty}* %ret
    ret void
  }
  """

  generated_items = []
  generated_metadata = []
  # FIXME: "cop" is missing
  for geom, vec, dtype, clamp in product(
      ["1d", "2d", "3d", "a1d", "a2d"],
      ["", "v2", "v4"],
      ["b8" , "b16", "b32", "b64"],
      ["trap", "clamp", "zero"]):

    if vec == "v4" and dtype == "b64":
      continue

    test_name = "test_suld_" + geom + vec + dtype + clamp

    params = {
      "test_name"   : test_name,

      "intrinsic"   : "llvm.nvvm.suld.{geom}.{dtype}.{clamp}".format(
        geom=get_llvm_geom(geom),
        dtype=(vec + get_llvm_type(dtype)),
        clamp=clamp),
      "retty"       : get_llvm_vec_type(vec, dtype),
      "access"      : get_llvm_surface_access(geom),
      "global_surf" : global_surf,

      "instruction" : "suld.b.{geom}{vec}.{dtype}.{clamp}".format(
        geom=geom,
        vec=("" if vec == "" else "." + vec),
        dtype=dtype,
        clamp=clamp),
      "reg_ret"     : get_ptx_vec_reg(vec, dtype),
      "reg_surf"    : get_ptx_surface(target),
      "reg_access"  : get_ptx_surface_access(geom),
    }
    gen_test(template, params)
    generated_items.append((params["intrinsic"], params["instruction"]))

    fun_name = test_name + "_param";
    fun_ty = "void (i64, {retty}*, {access_ty})*".format(
      retty=params["retty"],
      access_ty=get_llvm_surface_access_type(geom))
    generated_metadata += get_surface_metadata(
      target, fun_ty, fun_name, has_surface_param=True)

    fun_name = test_name + "_global";
    fun_ty = "void ({retty}*, {access_ty})*".format(
      retty=params["retty"],
      access_ty=get_llvm_surface_access_type(geom))
    generated_metadata += get_surface_metadata(
      target, fun_ty, fun_name, has_surface_param=False)

  return generated_items, generated_metadata

def gen_sust_tests(target, global_surf):
  """
  PTX spec s9.7.10.2. Surface Instructions

  sust.b.{1d,2d,3d}{.cop}.vec.ctype.clamp  [a, b], c;  // unformatted
  sust.p.{1d,2d,3d}.vec.b32.clamp          [a, b], c;  // formatted

  sust.b.{a1d,a2d}{.cop}.vec.ctype.clamp   [a, b], c;  // unformatted

  .cop   = { .wb, .cg, .cs, .wt };                     // cache operation
  .vec   = { none, .v2, .v4 };
  .ctype = { .b8 , .b16, .b32, .b64 };
  .clamp = { .trap, .clamp, .zero };
  """

  template = """
  declare void @${intrinsic}(i64 %s, ${access}, ${value});

  ; CHECK-LABEL: .entry ${test_name}_param
  ; CHECK: ${instruction} [${reg_surf}, ${reg_access}], ${reg_value}
  ;
  define void @${test_name}_param(i64 %s, ${value}, ${access}) {
    tail call void @${intrinsic}(i64 %s, ${access}, ${value})
    ret void
  }
  ; CHECK-LABEL: .entry ${test_name}_global
  ; CHECK: ${instruction} [${global_surf}, ${reg_access}], ${reg_value}
  ;
  define void @${test_name}_global(${value}, ${access}) {
    %gs = tail call i64 @llvm.nvvm.texsurf.handle.internal.p1i64(i64 addrspace(1)* @${global_surf})
    tail call void @${intrinsic}(i64 %gs, ${access}, ${value})
    ret void
  }
  """

  generated_items = []
  generated_metadata = []
  # FIXME: "cop" is missing
  for fmt, geom, vec, ctype, clamp in product(
      ["b", "p"],
      ["1d", "2d", "3d", "a1d", "a2d"],
      ["", "v2", "v4"],
      ["b8" , "b16", "b32", "b64"],
      ["trap", "clamp", "zero"]):

    if fmt == "p" and geom[0] == "a":
      continue
    if fmt == "p" and ctype != "b32":
      continue
    if vec == "v4" and ctype == "b64":
      continue

    # FIXME: these intrinsics are missing, but at least one of them is
    # listed in the PTX spec: sust.p.{1d,2d,3d}.vec.b32.clamp
    if fmt == "p" and clamp != "trap":
      continue

    test_name = "test_sust_" + fmt + geom + vec + ctype + clamp

    params = {
      "test_name"   : test_name,

      "intrinsic" : "llvm.nvvm.sust.{fmt}.{geom}.{ctype}.{clamp}".format(
        fmt=fmt,
        geom=get_llvm_geom(geom),
        ctype=(vec + get_llvm_type(ctype)),
        clamp=clamp),
      "access"      : get_llvm_surface_access(geom),
      "value"       : get_llvm_value(vec, ctype),
      "global_surf" : global_surf,

      "instruction" : "sust.{fmt}.{geom}{vec}.{ctype}.{clamp}".format(
        fmt=fmt,
        geom=geom,
        vec=("" if vec == "" else "." + vec),
        ctype=ctype,
        clamp=clamp),
      "reg_value"   : get_ptx_vec_reg(vec, ctype),
      "reg_surf"    : get_ptx_surface(target),
      "reg_access"  : get_ptx_surface_access(geom)
    }
    gen_test(template, params)
    generated_items.append((params["intrinsic"], params["instruction"]))

    fun_name = test_name + "_param";
    fun_ty = "void (i64, {value_ty}, {access_ty})*".format(
      value_ty=get_llvm_value_type(vec, ctype),
      access_ty=get_llvm_surface_access_type(geom))
    generated_metadata += get_surface_metadata(
      target, fun_ty, fun_name, has_surface_param=True)

    fun_name = test_name + "_global";
    fun_ty = "void ({value_ty}, {access_ty})*".format(
      value_ty=get_llvm_value_type(vec, ctype),
      access_ty=get_llvm_surface_access_type(geom))
    generated_metadata += get_surface_metadata(
      target, fun_ty, fun_name, has_surface_param=False)

  return generated_items, generated_metadata

def is_unified(target):
  """
  PTX has two modes of operation. In the unified mode, texture and
  sampler information is accessed through a single .texref handle. In
  the independent mode, texture and sampler information each have their
  own handle, allowing them to be defined separately and combined at the
  site of usage in the program.

  """
  return target == "cuda"

def get_llvm_texture_access(geom_ptx, ctype, mipmap):
  geom_access = {
    "1d"    : "{ctype} %x",
    "2d"    : "{ctype} %x, {ctype} %y",
    "3d"    : "{ctype} %x, {ctype} %y, {ctype} %z",
    "cube"  : "{ctype} %s, {ctype} %t, {ctype} %r",
    "a1d"   : "i32 %l, {ctype} %x",
    "a2d"   : "i32 %l, {ctype} %x, {ctype} %y",
    "acube" : "i32 %l, {ctype} %s, {ctype} %t, {ctype} %r",
  }

  access = geom_access[geom_ptx]

  if mipmap == "level":
    access += ", {ctype} %lvl"
  elif mipmap == "grad":
    if geom_ptx in ("1d", "a1d"):
      access += ", {ctype} %dpdx1, {ctype} %dpdy1"
    elif geom_ptx in ("2d", "a2d"):
      access += (", {ctype} %dpdx1, {ctype} %dpdx2" +
                 ", {ctype} %dpdy1, {ctype} %dpdy2")
    else:
      access += (", {ctype} %dpdx1, {ctype} %dpdx2, {ctype} %dpdx3" +
                 ", {ctype} %dpdy1, {ctype} %dpdy2, {ctype} %dpdy3")

  return access.format(ctype=get_llvm_type(ctype))

def get_llvm_texture_access_type(geom_ptx, ctype, mipmap):
  geom_access = {
    "1d"    : "{ctype}",
    "2d"    : "{ctype}, {ctype}",
    "3d"    : "{ctype}, {ctype}, {ctype}",
    "cube"  : "{ctype}, {ctype}, {ctype}",
    "a1d"   : "i32, {ctype}",
    "a2d"   : "i32, {ctype}, {ctype}",
    "acube" : "i32, {ctype}, {ctype}, {ctype}",
  }

  access = geom_access[geom_ptx]

  if mipmap == "level":
    access += ", {ctype}"
  elif mipmap == "grad":
    if geom_ptx in ("1d", "a1d"):
      access += ", {ctype}, {ctype}"
    elif geom_ptx in ("2d", "a2d"):
      access += (", {ctype}, {ctype}, {ctype}, {ctype}")
    else:
      access += (", {ctype}, {ctype}, {ctype}" +
                 ", {ctype}, {ctype}, {ctype}")

  return access.format(ctype=get_llvm_type(ctype))

def get_ptx_texture_access(geom_ptx, ctype):
  access_reg = {
    "1d"    : "{{{ctype_reg}}}",
    "2d"    : "{{{ctype_reg}, {ctype_reg}}}",
    "3d"    : "{{{ctype_reg}, {ctype_reg}, {ctype_reg}, {ctype_reg}}}",
    "a1d"   : "{{{b32_reg}, {ctype_reg}}}",
    "a2d"   : "{{{b32_reg}, {ctype_reg}, {ctype_reg}, {ctype_reg}}}",
    "cube"  : "{{{f32_reg}, {f32_reg}, {f32_reg}, {f32_reg}}}",
    "acube" : "{{{b32_reg}, {f32_reg}, {f32_reg}, {f32_reg}}}",
  }
  return access_reg[geom_ptx].format(ctype_reg=get_ptx_reg(ctype),
                                     b32_reg=get_ptx_reg("b32"),
                                     f32_reg=get_ptx_reg("f32"))

def get_ptx_texture(target):
  # With 'cuda' environment texture/sampler are copied with ld.param,
  # so the instruction uses registers. For 'nvcl' the instruction uses
  # texture/sampler parameters directly.
  if target == "cuda":
    return "%rd{{[0-9]+}}"
  elif target == "nvcl":
    return "test_{{.*}}_param_0, test_{{.*}}_param_1"
  raise RuntimeError("unknown target: " + target)

def get_llvm_global_sampler(target, global_sampler):
  if is_unified(target):
    return "", ""
  else:
    sampler_handle = "i64 %gs,"
    get_sampler_handle = (
      "%gs = tail call i64 @llvm.nvvm.texsurf.handle.internal.p1i64" +
      "(i64 addrspace(1)* @{})".format(global_sampler))
    return sampler_handle, get_sampler_handle

def get_ptx_global_sampler(target, global_sampler):
  if is_unified(target):
    return ""
  else:
    return global_sampler + ","

def get_texture_metadata(target, fun_ty, fun_name, has_texture_params):
  metadata = []

  md_kernel = "!{{{fun_ty} @{fun_name}, !\"kernel\", i32 1}}".format(
    fun_ty=fun_ty, fun_name=fun_name)
  metadata.append(md_kernel)

  if target == "cuda":
    # When a parameter is lowered as a .texref, it still has the
    # corresponding ld.param.u64, which is illegal. Do not emit the
    # metadata to keep the parameter as .b64 instead.
    has_texture_params = False

  if has_texture_params:
    md_texture = "!{{{fun_ty} @{fun_name}, !\"rdoimage\", i32 0}}".format(
      fun_ty=fun_ty, fun_name=fun_name)
    metadata.append(md_texture)

    if not is_unified(target):
      md_sampler = "!{{{fun_ty} @{fun_name}, !\"sampler\", i32 1}}".format(
      fun_ty=fun_ty, fun_name=fun_name)
      metadata.append(md_sampler)

  return metadata

def gen_tex_tests(target, global_tex, global_sampler):
  """
  PTX spec s9.7.9.3. Texture Instructions

  tex.geom.v4.dtype.ctype  d, [a, c] {, e} {, f};
  tex.geom.v4.dtype.ctype  d[|p], [a, b, c] {, e} {, f};  // explicit sampler

  tex.geom.v2.f16x2.ctype  d[|p], [a, c] {, e} {, f};
  tex.geom.v2.f16x2.ctype  d[|p], [a, b, c] {, e} {, f};  // explicit sampler

  // mipmaps
  tex.base.geom.v4.dtype.ctype   d[|p], [a, {b,} c] {, e} {, f};
  tex.level.geom.v4.dtype.ctype  d[|p], [a, {b,} c], lod {, e} {, f};
  tex.grad.geom.v4.dtype.ctype   d[|p], [a, {b,} c], dPdx, dPdy {, e} {, f};

  tex.base.geom.v2.f16x2.ctype   d[|p], [a, {b,} c] {, e} {, f};
  tex.level.geom.v2.f16x2.ctype  d[|p], [a, {b,} c], lod {, e} {, f};
  tex.grad.geom.v2.f16x2.ctype   d[|p], [a, {b,} c], dPdx, dPdy {, e} {, f};

  .geom  = { .1d, .2d, .3d, .a1d, .a2d, .cube, .acube, .2dms, .a2dms };
  .dtype = { .u32, .s32, .f16,  .f32 };
  .ctype = {       .s32, .f32 };          // .cube, .acube require .f32
                                          // .2dms, .a2dms require .s32
  """

  template = """
  declare ${retty} @${intrinsic}(i64 %tex, ${sampler} ${access})

  ; CHECK-LABEL: .entry ${test_name}_param
  ; CHECK: ${instruction} ${ptx_ret}, [${ptx_tex}, ${ptx_access}]
  define void @${test_name}_param(i64 %tex, ${sampler} ${retty}* %ret, ${access}) {
    %val = tail call ${retty} @${intrinsic}(i64 %tex, ${sampler} ${access})
    store ${retty} %val, ${retty}* %ret
    ret void
  }
  ; CHECK-LABEL: .entry ${test_name}_global
  ; CHECK: ${instruction} ${ptx_ret}, [${global_tex}, ${ptx_global_sampler} ${ptx_access}]
  define void @${test_name}_global(${retty}* %ret, ${access}) {
    %gt = tail call i64 @llvm.nvvm.texsurf.handle.internal.p1i64(i64 addrspace(1)* @${global_tex})
    ${get_sampler_handle}
    %val = tail call ${retty} @${intrinsic}(i64 %gt, ${sampler} ${access})
    store ${retty} %val, ${retty}* %ret
    ret void
  }
  """

  generated_items = []
  generated_metadata = []
  for mipmap, geom, vec, dtype, ctype in product(
      ["", "level", "grad"],
      ["1d", "2d", "3d", "a1d", "a2d", "cube", "acube", "2dms", "a2dms"],
      ["v2", "v4"],
      ["u32", "s32", "f16", "f32"],
      ["s32", "f32"]):

    # FIXME: missing intrinsics.
    # Multi-sample textures and multi-sample texture arrays
    # introduced in PTX ISA version 3.2.
    if geom in ("2dms", "a2dms"):
      continue

    # FIXME: missing intrinsics? no such restriction in the PTX spec
    if ctype == "s32" and mipmap != "":
      continue

    # FIXME: missing intrinsics?
    if ctype == "s32" and geom in ("cube", "acube"):
      continue

    # FIXME: missing intrinsics.
    # Support for textures returning f16 and f16x2 data introduced in
    # PTX ISA version 4.2.
    if vec == "v2" or dtype == "f16":
      continue

    # FIXME: missing intrinsics.
    # Support for tex.grad.{cube, acube} introduced in PTX ISA version
    # 4.3.
    if mipmap == "grad" and geom in ("cube", "acube"):
      continue

    # The instruction returns a two-element vector for destination
    # type f16x2. For all other destination types, the instruction
    # returns a four-element vector. Coordinates may be given in
    # either signed 32-bit integer or 32-bit floating point form.
    if vec == "v2" and dtype != "f16":
      continue

    sampler_handle, get_sampler_handle = get_llvm_global_sampler(
      target, global_sampler)

    test_name = "test_tex_" + "".join((mipmap, geom, vec, dtype, ctype))
    params = {
      "test_name" : test_name,
      "intrinsic" :
        "llvm.nvvm.tex{unified}.{geom}{mipmap}.{vec}{dtype}.{ctype}".format(
          unified=(".unified" if is_unified(target) else ""),
          geom=get_llvm_geom(geom),
          mipmap=("" if mipmap == "" else "." + mipmap),
          vec=vec,
          dtype=dtype,
          ctype=ctype),
      "global_tex": global_tex,
      "retty"     : get_llvm_vec_type(vec, dtype),
      "sampler"   : sampler_handle,
      "access"    : get_llvm_texture_access(geom, ctype, mipmap),
      "get_sampler_handle" : get_sampler_handle,

      "instruction" : "tex{mipmap}.{geom}.{vec}.{dtype}.{ctype}".format(
        mipmap=("" if mipmap == "" else "." + mipmap),
        geom=geom,
        vec=vec,
        dtype=dtype,
        ctype=ctype),
      "ptx_ret"     : get_ptx_vec_reg(vec, dtype),
      "ptx_tex"     : get_ptx_texture(target),
      "ptx_access"  : get_ptx_texture_access(geom, ctype),
      "ptx_global_sampler" : get_ptx_global_sampler(target, global_sampler),
    }
    gen_test(template, params)
    generated_items.append((params["intrinsic"], params["instruction"]))

    fun_name = test_name + "_param";
    fun_ty = "void (i64, {sampler} {retty}*, {access_ty})*".format(
      sampler=("" if is_unified(target) else "i64,"),
      retty=params["retty"],
      access_ty=get_llvm_texture_access_type(geom, ctype, mipmap))
    generated_metadata += get_texture_metadata(
      target, fun_ty, fun_name, has_texture_params=True)

    fun_name = test_name + "_global";
    fun_ty = "void ({retty}*, {access_ty})*".format(
      retty=params["retty"],
      access_ty=get_llvm_texture_access_type(geom, ctype, mipmap))
    generated_metadata += get_texture_metadata(
      target, fun_ty, fun_name, has_texture_params=False)

  return generated_items, generated_metadata

def get_llvm_tld4_access(geom):
  """
  For 2D textures, operand c specifies coordinates as a two-element,
  32-bit floating-point vector.

  For 2d texture arrays operand c is a four element, 32-bit
  vector. The first element in operand c is interpreted as an unsigned
  integer index (.u32) into the texture array, and the next two
  elements are interpreted as 32-bit floating point coordinates of 2d
  texture. The fourth element is ignored.

  For cubemap textures, operand c specifies four-element vector which
  comprises three floating-point coordinates (s, t, r) and a fourth
  padding argument which is ignored.

  [For cube arrays] The first element in operand c is interpreted as
  an unsigned integer index (.u32) into the cubemap texture array, and
  the remaining three elements are interpreted as floating-point
  cubemap coordinates (s, t, r), used to lookup in the selected
  cubemap.
  """
  geom_to_access = {
    "2d"    : "float %x, float %y",
    "a2d"   : "i32 %l, float %x, float %y",
    "cube"  : "float %s, float %t, float %r",
    "acube" : "i32 %l, float %s, float %t, float %r"
  }
  return geom_to_access[geom]

def get_llvm_tld4_access_type(geom):
  geom_to_access = {
    "2d"    : "float, float",
    "a2d"   : "i32, float, float",
    "cube"  : "float, float, float",
    "acube" : "i32, float, float, float"
  }
  return geom_to_access[geom]

def get_ptx_tld4_access(geom):
  geom_to_access = {
    "2d"    : "{%f{{[0-9]+}}, %f{{[0-9]+}}}",
    "a2d"   : "{%r{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}",
    "cube"  : "{%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}",
    "acube" : "{%r{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}"
  }
  return geom_to_access[geom]

def gen_tld4_tests(target, global_tex, global_sampler):
  """
  PTX spec s9.7.9.4. Texture Instructions: tld4
  Perform a texture fetch of the 4-texel bilerp footprint.

  tld4.comp.2d.v4.dtype.f32    d[|p], [a, c] {, e} {, f};
  tld4.comp.geom.v4.dtype.f32  d[|p], [a, b, c] {, e} {, f};  // explicit sampler

  .comp  = { .r, .g, .b, .a };
  .geom  = { .2d, .a2d, .cube, .acube };
  .dtype = { .u32, .s32, .f32 };
  """

  template = """
  declare ${retty} @${intrinsic}(i64 %tex, ${sampler} ${access})

  ; CHECK-LABEL: .entry ${test_name}_param
  ; CHECK: ${instruction} ${ptx_ret}, [${ptx_tex}, ${ptx_access}]
  define void @${test_name}_param(i64 %tex, ${sampler} ${retty}* %ret, ${access}) {
    %val = tail call ${retty} @${intrinsic}(i64 %tex, ${sampler} ${access})
    store ${retty} %val, ${retty}* %ret
    ret void
  }
  ; CHECK-LABEL: .entry ${test_name}_global
  ; CHECK: ${instruction} ${ptx_ret}, [${global_tex}, ${ptx_global_sampler} ${ptx_access}]
  define void @${test_name}_global(${retty}* %ret, ${access}) {
    %gt = tail call i64 @llvm.nvvm.texsurf.handle.internal.p1i64(i64 addrspace(1)* @${global_tex})
    ${get_sampler_handle}
    %val = tail call ${retty} @${intrinsic}(i64 %gt, ${sampler} ${access})
    store ${retty} %val, ${retty}* %ret
    ret void
  }
  """

  generated_items = []
  generated_metadata = []
  for comp, geom, dtype in product(
      ["r", "g", "b", "a"],
      ["2d", "a2d", "cube", "acube"],
      ["u32", "s32", "f32"]):

    # FIXME: missing intrinsics.
    # tld4.{a2d,cube,acube} introduced in PTX ISA version 4.3.
    if geom in ("a2d", "cube", "acube"):
      continue

    sampler_handle, get_sampler_handle = get_llvm_global_sampler(
      target, global_sampler)

    test_name = "test_tld4_" + "".join((comp, geom, dtype))
    params = {
      "test_name" : test_name,
      "intrinsic" :
        "llvm.nvvm.tld4{unified}.{comp}.{geom}.v4{dtype}.f32".format(
          unified=(".unified" if is_unified(target) else ""),
          comp=comp,
          geom=get_llvm_geom(geom),
          dtype=dtype),
      "global_tex" : global_tex,
      "retty"      : get_llvm_vec_type("v4", dtype),
      "sampler"    : sampler_handle,
      "access"     : get_llvm_tld4_access(geom),
      "get_sampler_handle" : get_sampler_handle,

      "instruction" : "tld4.{comp}.{geom}.v4.{dtype}.f32".format(
        comp=comp, geom=geom, dtype=dtype),
      "ptx_ret"     : get_ptx_vec_reg("v4", dtype),
      "ptx_tex"     : get_ptx_texture(target),
      "ptx_access"  : get_ptx_tld4_access(geom),
      "ptx_global_sampler" : get_ptx_global_sampler(target, global_sampler),
    }
    gen_test(template, params)
    generated_items.append((params["intrinsic"], params["instruction"]))

    fun_name = test_name + "_param";
    fun_ty = "void (i64, {sampler} {retty}*, {access_ty})*".format(
      sampler=("" if is_unified(target) else "i64,"),
      retty=params["retty"],
      access_ty=get_llvm_tld4_access_type(geom))
    generated_metadata += get_texture_metadata(
      target, fun_ty, fun_name, has_texture_params=True)

    fun_name = test_name + "_global";
    fun_ty = "void ({retty}*, {access_ty})*".format(
      retty=params["retty"],
      access_ty=get_llvm_tld4_access_type(geom))
    generated_metadata += get_texture_metadata(
      target, fun_ty, fun_name, has_texture_params=False)

  return generated_items, generated_metadata

def gen_test(template, params):
  if debug:
    print()
    for param, value in params.items():
      print(";; {}: {}".format(param, value))

  print(string.Template(textwrap.dedent(template)).substitute(params))

def gen_tests(target, tests):
  gen_triple(target)

  items = []
  metadata = []

  global_surf = "gsurf"
  global_tex = "gtex"
  global_sampler = "gsam"
  metadata += gen_globals(target, global_surf, global_tex, global_sampler)

  if "suld" in tests:
    suld_items, suld_md = gen_suld_tests(target, global_surf)
    items += suld_items
    metadata += suld_md
  if "sust" in tests:
    sust_items, sust_md = gen_sust_tests(target, global_surf)
    items += sust_items
    metadata += sust_md
  if "tex" in tests:
    tex_items, tex_md = gen_tex_tests(target, global_tex, global_sampler)
    items += tex_items
    metadata += tex_md
  if "tld4" in tests:
    tld4_items, tld4_md = gen_tld4_tests(target, global_tex, global_sampler)
    items += tld4_items
    metadata += tld4_md

  gen_metadata(metadata)
  return items

def write_gen_list(filename, append, items):
  with open(filename, ("a" if append else "w")) as f:
    for intrinsic, instruction in items:
      f.write("{} {}\n".format(intrinsic, instruction))

def read_gen_list(filename):
  intrinsics = set()
  instructions = set()
  with open(filename) as f:
    for line in f:
      intrinsic, instruction = line.split()
      intrinsics.add(intrinsic)
      instructions.add(instruction)
  return (intrinsics, instructions)

def read_td_list(filename, regex):
  td_list = set()
  with open(filename) as f:
    for line in f:
      match = re.search(regex, line)
      if match:
        td_list.add(match.group(1))

  # Arbitrary value - we should find quite a lot of instructions
  if len(td_list) < 30:
    raise RuntimeError("found only {} instructions in {}".format(
      filename, len(td_list)))

  return td_list

def verify_inst_tablegen(path_td, gen_instr):
  """
  Verify that all instructions defined in NVPTXIntrinsics.td are
  tested.
  """

  td_instr = read_td_list(path_td, "\"((suld|sust|tex|tld4)\\..*)\"")

  gen_instr.update({
    # FIXME: spec does not list any sust.p variants other than b32
    "sust.p.1d.b8.trap",
    "sust.p.1d.b16.trap",
    "sust.p.1d.v2.b8.trap",
    "sust.p.1d.v2.b16.trap",
    "sust.p.1d.v4.b8.trap",
    "sust.p.1d.v4.b16.trap",
    "sust.p.a1d.b8.trap",
    "sust.p.a1d.b16.trap",
    "sust.p.a1d.v2.b8.trap",
    "sust.p.a1d.v2.b16.trap",
    "sust.p.a1d.v4.b8.trap",
    "sust.p.a1d.v4.b16.trap",
    "sust.p.2d.b8.trap",
    "sust.p.2d.b16.trap",
    "sust.p.2d.v2.b8.trap",
    "sust.p.2d.v2.b16.trap",
    "sust.p.2d.v4.b8.trap",
    "sust.p.2d.v4.b16.trap",
    "sust.p.a2d.b8.trap",
    "sust.p.a2d.b16.trap",
    "sust.p.a2d.v2.b8.trap",
    "sust.p.a2d.v2.b16.trap",
    "sust.p.a2d.v4.b8.trap",
    "sust.p.a2d.v4.b16.trap",
    "sust.p.3d.b8.trap",
    "sust.p.3d.b16.trap",
    "sust.p.3d.v2.b8.trap",
    "sust.p.3d.v2.b16.trap",
    "sust.p.3d.v4.b8.trap",
    "sust.p.3d.v4.b16.trap",

    # FIXME: sust.p is also not supported for arrays
    "sust.p.a1d.b32.trap",
    "sust.p.a1d.v2.b32.trap",
    "sust.p.a1d.v4.b32.trap",
    "sust.p.a2d.b32.trap",
    "sust.p.a2d.v2.b32.trap",
    "sust.p.a2d.v4.b32.trap",
  })

  td_instr = list(td_instr)
  td_instr.sort()
  gen_instr = list(gen_instr)
  gen_instr.sort()
  for i, td in enumerate(td_instr):
    if i == len(gen_instr) or td != gen_instr[i]:
      raise RuntimeError(
        "{} is present in tablegen, but not tested.\n".format(td))

def verify_llvm_tablegen(path_td, gen_intr):
  """
  Verify that all intrinsics defined in IntrinsicsNVVM.td are
  tested.
  """

  td_intr = read_td_list(
    path_td, "\"(llvm\\.nvvm\\.(suld|sust|tex|tld4)\\..*)\"")

  gen_intr.update({
    # FIXME: spec does not list any sust.p variants other than b32
    "llvm.nvvm.sust.p.1d.i8.trap",
    "llvm.nvvm.sust.p.1d.i16.trap",
    "llvm.nvvm.sust.p.1d.v2i8.trap",
    "llvm.nvvm.sust.p.1d.v2i16.trap",
    "llvm.nvvm.sust.p.1d.v4i8.trap",
    "llvm.nvvm.sust.p.1d.v4i16.trap",
    "llvm.nvvm.sust.p.1d.array.i8.trap",
    "llvm.nvvm.sust.p.1d.array.i16.trap",
    "llvm.nvvm.sust.p.1d.array.v2i8.trap",
    "llvm.nvvm.sust.p.1d.array.v2i16.trap",
    "llvm.nvvm.sust.p.1d.array.v4i8.trap",
    "llvm.nvvm.sust.p.1d.array.v4i16.trap",
    "llvm.nvvm.sust.p.2d.i8.trap",
    "llvm.nvvm.sust.p.2d.i16.trap",
    "llvm.nvvm.sust.p.2d.v2i8.trap",
    "llvm.nvvm.sust.p.2d.v2i16.trap",
    "llvm.nvvm.sust.p.2d.v4i8.trap",
    "llvm.nvvm.sust.p.2d.v4i16.trap",
    "llvm.nvvm.sust.p.2d.array.i8.trap",
    "llvm.nvvm.sust.p.2d.array.i16.trap",
    "llvm.nvvm.sust.p.2d.array.v2i8.trap",
    "llvm.nvvm.sust.p.2d.array.v2i16.trap",
    "llvm.nvvm.sust.p.2d.array.v4i8.trap",
    "llvm.nvvm.sust.p.2d.array.v4i16.trap",
    "llvm.nvvm.sust.p.3d.i8.trap",
    "llvm.nvvm.sust.p.3d.i16.trap",
    "llvm.nvvm.sust.p.3d.v2i8.trap",
    "llvm.nvvm.sust.p.3d.v2i16.trap",
    "llvm.nvvm.sust.p.3d.v4i8.trap",
    "llvm.nvvm.sust.p.3d.v4i16.trap",

    # FIXME: sust.p is also not supported for arrays
    "llvm.nvvm.sust.p.1d.array.i32.trap",
    "llvm.nvvm.sust.p.1d.array.v2i32.trap",
    "llvm.nvvm.sust.p.1d.array.v4i32.trap",
    "llvm.nvvm.sust.p.2d.array.i32.trap",
    "llvm.nvvm.sust.p.2d.array.v2i32.trap",
    "llvm.nvvm.sust.p.2d.array.v4i32.trap"
  })

  td_intr = list(td_intr)
  td_intr.sort()
  gen_intr = list(gen_intr)
  gen_intr.sort()
  for i, td in enumerate(td_intr):
    if i == len(gen_intr) or td != gen_intr[i]:
      raise RuntimeError(
        "{} is present in tablegen, but not tested.\n".format(td))

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true")
parser.add_argument("--tests", type=str)
parser.add_argument("--target", type=str)
parser.add_argument("--gen-list", dest="gen_list", type=str)
parser.add_argument("--gen-list-append", dest="gen_list_append",
                    action="store_true")
parser.add_argument("--verify", action="store_true")
parser.add_argument("--llvm-tablegen", dest="llvm_td", type=str)
parser.add_argument("--inst-tablegen", dest="inst_td", type=str)

args = parser.parse_args()
debug = args.debug

if args.verify:
  intrinsics, instructions = read_gen_list(args.gen_list)
  verify_inst_tablegen(args.inst_td, instructions)
  verify_llvm_tablegen(args.llvm_td, intrinsics)
else:
  items = gen_tests(args.target, args.tests.split(","))
  if (args.gen_list):
    write_gen_list(args.gen_list, args.gen_list_append, items)
