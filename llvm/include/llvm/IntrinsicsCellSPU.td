//==- IntrinsicsCellSPU.td - Cell SDK intrinsics           -*- tablegen -*-==//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// Department at The Aerospace Corporation and is distributed under the
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// Cell SPU Instructions:
//===----------------------------------------------------------------------===//
// TODO Items (not urgent today, but would be nice, low priority)
//
// ANDBI, ORBI: SPU constructs a 4-byte constant for these instructions by
// concatenating the byte argument b as "bbbb". Could recognize this bit pattern
// in 16-bit and 32-bit constants and reduce instruction count.
//===----------------------------------------------------------------------===//

// 7-bit integer type, used as an immediate:
def cell_i7_ty: LLVMType<i8>;
def cell_i8_ty: LLVMType<i8>;

// Keep this here until it's actually supported:
def llvm_i128_ty : LLVMType<i128>;

class v16i8_u7imm<string builtin_suffix> :
  GCCBuiltin<!strconcat("__builtin_si_", builtin_suffix)>,
  Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, cell_i7_ty],
            [IntrNoMem]>;

class v16i8_u8imm<string builtin_suffix> :
  GCCBuiltin<!strconcat("__builtin_si_", builtin_suffix)>,
  Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_i8_ty],
            [IntrNoMem]>;

class v16i8_s10imm<string builtin_suffix> :
  GCCBuiltin<!strconcat("__builtin_si_", builtin_suffix)>,
  Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_i16_ty],
            [IntrNoMem]>;

class v16i8_u16imm<string builtin_suffix> :
  GCCBuiltin<!strconcat("__builtin_si_", builtin_suffix)>,
  Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_i16_ty],
            [IntrNoMem]>;

class v16i8_rr<string builtin_suffix> :
  GCCBuiltin<!strconcat("__builtin_si_", builtin_suffix)>,
  Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_v16i8_ty],
            [IntrNoMem]>;

class v8i16_s10imm<string builtin_suffix> :
  GCCBuiltin<!strconcat("__builtin_si_", builtin_suffix)>,
  Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_i16_ty],
            [IntrNoMem]>;

class v8i16_u16imm<string builtin_suffix> :
  GCCBuiltin<!strconcat("__builtin_si_", builtin_suffix)>,
  Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_i16_ty],
            [IntrNoMem]>;

class v8i16_rr<string builtin_suffix> :
  GCCBuiltin<!strconcat("__builtin_si_", builtin_suffix)>,
  Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
            [IntrNoMem]>;

class v4i32_rr<string builtin_suffix> :
  GCCBuiltin<!strconcat("__builtin_si_", builtin_suffix)>,
  Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
            [IntrNoMem]>;

class v4i32_u7imm<string builtin_suffix> :
  GCCBuiltin<!strconcat("__builtin_si_", builtin_suffix)>,
  Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, cell_i7_ty],
            [IntrNoMem]>;

class v4i32_s10imm<string builtin_suffix> :
  GCCBuiltin<!strconcat("__builtin_si_", builtin_suffix)>,
  Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_i16_ty],
            [IntrNoMem]>;

class v4i32_u16imm<string builtin_suffix> :
  GCCBuiltin<!strconcat("__builtin_si_", builtin_suffix)>,
  Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_i16_ty],
            [IntrNoMem]>;

class v4f32_rr<string builtin_suffix> :
  GCCBuiltin<!strconcat("__builtin_si_", builtin_suffix)>,
  Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty],
            [IntrNoMem]>;

class v4f32_rrr<string builtin_suffix> :
  GCCBuiltin<!strconcat("__builtin_si_", builtin_suffix)>,
  Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty],
            [IntrNoMem]>;

class v2f64_rr<string builtin_suffix> :
  GCCBuiltin<!strconcat("__builtin_si_", builtin_suffix)>,
  Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty],
            [IntrNoMem]>;

// All Cell SPU intrinsics start with "llvm.spu.".
let TargetPrefix = "spu" in {
  def int_spu_si_fsmbi  : v8i16_u16imm<"fsmbi">;
  def int_spu_si_ah     : v8i16_rr<"ah">;
  def int_spu_si_ahi    : v8i16_s10imm<"ahi">;
  def int_spu_si_a      : v4i32_rr<"a">;
  def int_spu_si_ai     : v4i32_s10imm<"ai">;
  def int_spu_si_sfh    : v8i16_rr<"sfh">;
  def int_spu_si_sfhi   : v8i16_s10imm<"sfhi">;
  def int_spu_si_sf     : v4i32_rr<"sf">;
  def int_spu_si_sfi    : v4i32_s10imm<"sfi">;
  def int_spu_si_addx   : v4i32_rr<"addx">;
  def int_spu_si_cg     : v4i32_rr<"cg">;
  def int_spu_si_cgx    : v4i32_rr<"cgx">;
  def int_spu_si_sfx    : v4i32_rr<"sfx">;
  def int_spu_si_bg     : v4i32_rr<"bg">;
  def int_spu_si_bgx    : v4i32_rr<"bgx">;
  def int_spu_si_mpy    : // This is special:
    GCCBuiltin<"__builtin_si_mpy">,
    Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
              [IntrNoMem]>;
  def int_spu_si_mpyu   : // This is special:
    GCCBuiltin<"__builtin_si_mpyu">,
    Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
              [IntrNoMem]>;
  def int_spu_si_mpyi   : // This is special:
    GCCBuiltin<"__builtin_si_mpyi">,
    Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_i16_ty],
              [IntrNoMem]>;
  def int_spu_si_mpyui  : // This is special:
    GCCBuiltin<"__builtin_si_mpyui">,
    Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_i16_ty],
              [IntrNoMem]>;
  def int_spu_si_mpya   : // This is special:
    GCCBuiltin<"__builtin_si_mpya">,
    Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty],
              [IntrNoMem]>;
  def int_spu_si_mpyh   : // This is special:
    GCCBuiltin<"__builtin_si_mpyh">,
    Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v8i16_ty],
              [IntrNoMem]>;
  def int_spu_si_mpys   : // This is special:
    GCCBuiltin<"__builtin_si_mpys">,
    Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
              [IntrNoMem]>;
  def int_spu_si_mpyhh  : // This is special:
    GCCBuiltin<"__builtin_si_mpyhh">,
    Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
              [IntrNoMem]>;
  def int_spu_si_mpyhha : // This is special:
    GCCBuiltin<"__builtin_si_mpyhha">,
    Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
              [IntrNoMem]>;
  def int_spu_si_mpyhhu : // This is special:
    GCCBuiltin<"__builtin_si_mpyhhu">,
    Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
              [IntrNoMem]>;
  def int_spu_si_mpyhhau : // This is special:
    GCCBuiltin<"__builtin_si_mpyhhau">,
    Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
              [IntrNoMem]>;

  def int_spu_si_shli:          v4i32_u7imm<"shli">;

  def int_spu_si_shlqbi:
    GCCBuiltin<!strconcat("__builtin_si_", "shlqbi")>,
    Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_i32_ty],
              [IntrNoMem]>;

  def int_spu_si_shlqbii:       v16i8_u7imm<"shlqbii">;
  def int_spu_si_shlqby:
    GCCBuiltin<!strconcat("__builtin_si_", "shlqby")>,
    Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_i32_ty],
              [IntrNoMem]>;
  def int_spu_si_shlqbyi:       v16i8_u7imm<"shlqbyi">;
  
  def int_spu_si_ceq:           v4i32_rr<"ceq">;
  def int_spu_si_ceqi:          v4i32_s10imm<"ceqi">;
  def int_spu_si_ceqb:          v16i8_rr<"ceqb">;
  def int_spu_si_ceqbi:         v16i8_u8imm<"ceqbi">;
  def int_spu_si_ceqh:          v8i16_rr<"ceqh">;
  def int_spu_si_ceqhi:         v8i16_s10imm<"ceqhi">;
  def int_spu_si_cgt:           v4i32_rr<"cgt">;
  def int_spu_si_cgti:          v4i32_s10imm<"cgti">;
  def int_spu_si_cgtb:          v16i8_rr<"cgtb">;
  def int_spu_si_cgtbi:         v16i8_u8imm<"cgtbi">;
  def int_spu_si_cgth:          v8i16_rr<"cgth">;
  def int_spu_si_cgthi:         v8i16_s10imm<"cgthi">;
  def int_spu_si_clgtb:         v16i8_rr<"clgtb">;
  def int_spu_si_clgtbi:        v16i8_u8imm<"clgtbi">;
  def int_spu_si_clgth:         v8i16_rr<"clgth">;
  def int_spu_si_clgthi:        v8i16_s10imm<"clgthi">;
  def int_spu_si_clgt:          v4i32_rr<"clgt">;
  def int_spu_si_clgti:         v4i32_s10imm<"clgti">;
  
  def int_spu_si_and:           v4i32_rr<"and">;
  def int_spu_si_andbi:         v16i8_u8imm<"andbi">;
  def int_spu_si_andc:          v4i32_rr<"andc">;
  def int_spu_si_andhi:         v8i16_s10imm<"andhi">;
  def int_spu_si_andi:          v4i32_s10imm<"andi">;
  
  def int_spu_si_or:            v4i32_rr<"or">;
  def int_spu_si_orbi:          v16i8_u8imm<"orbi">;
  def int_spu_si_orc:           v4i32_rr<"orc">;
  def int_spu_si_orhi:          v8i16_s10imm<"orhi">;
  def int_spu_si_ori:           v4i32_s10imm<"ori">;
  
  def int_spu_si_xor:           v4i32_rr<"xor">;
  def int_spu_si_xorbi:         v16i8_u8imm<"xorbi">;
  def int_spu_si_xorhi:         v8i16_s10imm<"xorhi">;
  def int_spu_si_xori:          v4i32_s10imm<"xori">;

  def int_spu_si_nor:           v4i32_rr<"nor">;
  def int_spu_si_nand:          v4i32_rr<"nand">;
  
  def int_spu_si_fa:            v4f32_rr<"fa">;
  def int_spu_si_fs:            v4f32_rr<"fs">;
  def int_spu_si_fm:            v4f32_rr<"fm">;
  
  def int_spu_si_fceq:          v4f32_rr<"fceq">;
  def int_spu_si_fcmeq:         v4f32_rr<"fcmeq">;
  def int_spu_si_fcgt:          v4f32_rr<"fcgt">;
  def int_spu_si_fcmgt:         v4f32_rr<"fcmgt">;
  
  def int_spu_si_fma:           v4f32_rrr<"fma">;
  def int_spu_si_fnms:          v4f32_rrr<"fnms">;
  def int_spu_si_fms:           v4f32_rrr<"fms">;

  def int_spu_si_dfa:           v2f64_rr<"dfa">;
  def int_spu_si_dfs:           v2f64_rr<"dfs">;
  def int_spu_si_dfm:           v2f64_rr<"dfm">;
  
//def int_spu_si_dfceq:         v2f64_rr<"dfceq">;
//def int_spu_si_dfcmeq:        v2f64_rr<"dfcmeq">;
//def int_spu_si_dfcgt:         v2f64_rr<"dfcgt">;
//def int_spu_si_dfcmgt:        v2f64_rr<"dfcmgt">;
  
  def int_spu_si_dfnma:         v2f64_rr<"dfnma">;
  def int_spu_si_dfma:          v2f64_rr<"dfma">;
  def int_spu_si_dfnms:         v2f64_rr<"dfnms">;
  def int_spu_si_dfms:          v2f64_rr<"dfms">;
}
