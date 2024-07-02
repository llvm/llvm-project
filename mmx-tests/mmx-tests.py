#!/usr/bin/python3

import argparse
import sys

# This is a list of all intel functions and macros which take or
# return an __m64.
def do_mmx(fn):
  # mmintrin.h
  fn("_mm_cvtsi32_si64", "__m64", ("int", ))
  fn("_mm_cvtsi64_si32", "int", ("__m64", ))
  fn("_mm_cvtsi64_m64", "__m64", ("long long", ), condition='defined(__X86_64__) || defined(__clang__)')
  fn("_mm_cvtm64_si64", "long long", ("__m64", ), condition='defined(__X86_64__) || defined(__clang__)')
  fn("_mm_packs_pi16", "__m64", ("__m64", "__m64", ))
  fn("_mm_packs_pi32", "__m64", ("__m64", "__m64", ))
  fn("_mm_packs_pu16", "__m64", ("__m64", "__m64", ))
  fn("_mm_unpackhi_pi8", "__m64", ("__m64", "__m64", ))
  fn("_mm_unpackhi_pi16", "__m64", ("__m64", "__m64", ))
  fn("_mm_unpackhi_pi32", "__m64", ("__m64", "__m64", ))
  fn("_mm_unpacklo_pi8", "__m64", ("__m64", "__m64", ))
  fn("_mm_unpacklo_pi16", "__m64", ("__m64", "__m64", ))
  fn("_mm_unpacklo_pi32", "__m64", ("__m64", "__m64", ))
  fn("_mm_add_pi8", "__m64", ("__m64", "__m64", ))
  fn("_mm_add_pi16", "__m64", ("__m64", "__m64", ))
  fn("_mm_add_pi32", "__m64", ("__m64", "__m64", ))
  fn("_mm_adds_pi8", "__m64", ("__m64", "__m64", ))
  fn("_mm_adds_pi16", "__m64", ("__m64", "__m64", ))
  fn("_mm_adds_pu8", "__m64", ("__m64", "__m64", ))
  fn("_mm_adds_pu16", "__m64", ("__m64", "__m64", ))
  fn("_mm_sub_pi8", "__m64", ("__m64", "__m64", ))
  fn("_mm_sub_pi16", "__m64", ("__m64", "__m64", ))
  fn("_mm_sub_pi32", "__m64", ("__m64", "__m64", ))
  fn("_mm_subs_pi8", "__m64", ("__m64", "__m64", ))
  fn("_mm_subs_pi16", "__m64", ("__m64", "__m64", ))
  fn("_mm_subs_pu8", "__m64", ("__m64", "__m64", ))
  fn("_mm_subs_pu16", "__m64", ("__m64", "__m64", ))
  fn("_mm_madd_pi16", "__m64", ("__m64", "__m64", ))
  fn("_mm_mulhi_pi16", "__m64", ("__m64", "__m64", ))
  fn("_mm_mullo_pi16", "__m64", ("__m64", "__m64", ))
  fn("_mm_sll_pi16", "__m64", ("__m64", "__m64", ))
  fn("_mm_slli_pi16", "__m64", ("__m64", "int", ))
  fn("_mm_sll_pi32", "__m64", ("__m64", "__m64", ))
  fn("_mm_slli_pi32", "__m64", ("__m64", "int", ))
  fn("_mm_sll_si64", "__m64", ("__m64", "__m64", ))
  fn("_mm_slli_si64", "__m64", ("__m64", "int", ))
  fn("_mm_sra_pi16", "__m64", ("__m64", "__m64", ))
  fn("_mm_srai_pi16", "__m64", ("__m64", "int", ))
  fn("_mm_sra_pi32", "__m64", ("__m64", "__m64", ))
  fn("_mm_srai_pi32", "__m64", ("__m64", "int", ))
  fn("_mm_srl_pi16", "__m64", ("__m64", "__m64", ))
  fn("_mm_srli_pi16", "__m64", ("__m64", "int", ))
  fn("_mm_srl_pi32", "__m64", ("__m64", "__m64", ))
  fn("_mm_srli_pi32", "__m64", ("__m64", "int", ))
  fn("_mm_srl_si64", "__m64", ("__m64", "__m64", ))
  fn("_mm_srli_si64", "__m64", ("__m64", "int", ))
  fn("_mm_and_si64", "__m64", ("__m64", "__m64", ))
  fn("_mm_andnot_si64", "__m64", ("__m64", "__m64", ))
  fn("_mm_or_si64", "__m64", ("__m64", "__m64", ))
  fn("_mm_xor_si64", "__m64", ("__m64", "__m64", ))
  fn("_mm_cmpeq_pi8", "__m64", ("__m64", "__m64", ))
  fn("_mm_cmpeq_pi16", "__m64", ("__m64", "__m64", ))
  fn("_mm_cmpeq_pi32", "__m64", ("__m64", "__m64", ))
  fn("_mm_cmpgt_pi8", "__m64", ("__m64", "__m64", ))
  fn("_mm_cmpgt_pi16", "__m64", ("__m64", "__m64", ))
  fn("_mm_cmpgt_pi32", "__m64", ("__m64", "__m64", ))
  fn("_mm_setzero_si64", "__m64", ())
  fn("_mm_set_pi32", "__m64", ("int", "int", ))
  fn("_mm_set_pi16", "__m64", ("short", "short", "short", "short", ))
  fn("_mm_set_pi8", "__m64", ("char", "char", "char", "char", "char", "char", "char", "char", ))
  fn("_mm_set1_pi32", "__m64", ("int", ))
  fn("_mm_set1_pi16", "__m64", ("short", ))
  fn("_mm_set1_pi8", "__m64", ("char", ))
  fn("_mm_setr_pi32", "__m64", ("int", "int", ))
  fn("_mm_setr_pi16", "__m64", ("short", "short", "short", "short", ))
  fn("_mm_setr_pi8", "__m64", ("char", "char", "char", "char", "char", "char", "char", "char", ))

  # xmmintrin.h
  fn("_mm_cvtps_pi32", "__m64", ("__m128", ))
  fn("_mm_cvt_ps2pi", "__m64", ("__m128", ))
  fn("_mm_cvttps_pi32", "__m64", ("__m128", ))
  fn("_mm_cvtt_ps2pi", "__m64", ("__m128", ))
  fn("_mm_cvtpi32_ps", "__m128", ("__m128", "__m64", ))
  fn("_mm_cvt_pi2ps", "__m128", ("__m128", "__m64", ))
  fn("_mm_loadh_pi", "__m128", ("__m128", "const __m64 *", ))
  fn("_mm_loadl_pi", "__m128", ("__m128", "const __m64 *", ))
  fn("_mm_storeh_pi", "void", ("__m64 *", "__m128", ))
  fn("_mm_storel_pi", "void", ("__m64 *", "__m128", ))
  fn("_mm_stream_pi", "void", ("__m64 *", "__m64", ))
  fn("_mm_max_pi16", "__m64", ("__m64", "__m64", ))
  fn("_mm_max_pu8", "__m64", ("__m64", "__m64", ))
  fn("_mm_min_pi16", "__m64", ("__m64", "__m64", ))
  fn("_mm_min_pu8", "__m64", ("__m64", "__m64", ))
  fn("_mm_movemask_pi8", "int", ("__m64", ))
  fn("_mm_mulhi_pu16", "__m64", ("__m64", "__m64", ))
  fn("_mm_maskmove_si64", "void", ("__m64", "__m64", "char *", ))
  fn("_mm_avg_pu8", "__m64", ("__m64", "__m64", ))
  fn("_mm_avg_pu16", "__m64", ("__m64", "__m64", ))
  fn("_mm_sad_pu8", "__m64", ("__m64", "__m64", ))
  fn("_mm_cvtpi16_ps", "__m128", ("__m64", ))
  fn("_mm_cvtpu16_ps", "__m128", ("__m64", ))
  fn("_mm_cvtpi8_ps", "__m128", ("__m64", ))
  fn("_mm_cvtpu8_ps", "__m128", ("__m64", ))
  fn("_mm_cvtpi32x2_ps", "__m128", ("__m64", "__m64", ))
  fn("_mm_cvtps_pi16", "__m64", ("__m128", ))
  fn("_mm_cvtps_pi8", "__m64", ("__m128", ))

  fn("_mm_extract_pi16", "int", ("__m64", "int", ), imm_range=(0, 3))
  fn("_mm_insert_pi16", "__m64", ("__m64", "int", "int", ), imm_range=(0, 3))
  fn("_mm_shuffle_pi16", "__m64", ("__m64", "int", ), imm_range=(0, 255))

  # emmintrin.h
  fn("_mm_cvtpd_pi32", "__m64", ("__m128d", ))
  fn("_mm_cvttpd_pi32", "__m64", ("__m128d", ))
  fn("_mm_cvtpi32_pd", "__m128d", ("__m64", ))
  fn("_mm_add_si64", "__m64", ("__m64", "__m64", ))
  fn("_mm_mul_su32", "__m64", ("__m64", "__m64", ))
  fn("_mm_sub_si64", "__m64", ("__m64", "__m64", ))
  fn("_mm_set_epi64", "__m128i", ("__m64", "__m64", ))
  fn("_mm_set1_epi64", "__m128i", ("__m64", ))
  fn("_mm_setr_epi64", "__m128i", ("__m64", "__m64", ))
  fn("_mm_movepi64_pi64", "__m64", ("__m128i", ))
  fn("_mm_movpi64_epi64", "__m128i", ("__m64", ))

  # tmmintrin.h
  fn("_mm_abs_pi8", "__m64", ("__m64", ), target='ssse3')
  fn("_mm_abs_pi16", "__m64", ("__m64", ), target='ssse3')
  fn("_mm_abs_pi32", "__m64", ("__m64", ), target='ssse3')
  fn("_mm_hadd_pi16", "__m64", ("__m64", "__m64", ), target='ssse3')
  fn("_mm_hadd_pi32", "__m64", ("__m64", "__m64", ), target='ssse3')
  fn("_mm_hadds_pi16", "__m64", ("__m64", "__m64", ), target='ssse3')
  fn("_mm_hsub_pi16", "__m64", ("__m64", "__m64", ), target='ssse3')
  fn("_mm_hsub_pi32", "__m64", ("__m64", "__m64", ), target='ssse3')
  fn("_mm_hsubs_pi16", "__m64", ("__m64", "__m64", ), target='ssse3')
  fn("_mm_maddubs_pi16", "__m64", ("__m64", "__m64", ), target='ssse3')
  fn("_mm_mulhrs_pi16", "__m64", ("__m64", "__m64", ), target='ssse3')
  fn("_mm_shuffle_pi8", "__m64", ("__m64", "__m64", ), target='ssse3')
  fn("_mm_sign_pi8", "__m64", ("__m64", "__m64", ), target='ssse3')
  fn("_mm_sign_pi16", "__m64", ("__m64", "__m64", ), target='ssse3')
  fn("_mm_sign_pi32", "__m64", ("__m64", "__m64", ), target='ssse3')
  fn("_mm_alignr_pi8", "__m64", ("__m64", "__m64", "int", ), imm_range=(0, 18), target='ssse3')

# Generate a file full of wrapper functions for each of the above mmx
# functions.
#
# If use_xmm is set, pass/return arguments as __m128 rather than of
# __m64.
def define_wrappers(prefix, use_xmm=True, header=False):
  if header:
    print('#pragma once')

  print('#include <immintrin.h>')
  if use_xmm and not header:
    print('#define m128_to_m64(x) ((__m64)((__v2di)(x))[0])')
    print('#define m64_to_m128(x) ((__m128)(__v2di){(long long)(__m64)(x), 0})')

  def fn(name, ret_ty, arg_tys, imm_range=None, target=None, condition=None):
    if condition:
      print(f'#if {condition}')
    convert_ret = False
    if use_xmm and ret_ty == '__m64':
      ret_ty = '__v2di'
      convert_ret = True

    if target:
      attr = f'__attribute__((target("{target}"))) '
    else:
      attr = ''

    if imm_range:
      arg_tys = arg_tys[:-1]
    def translate_type(t):
      if use_xmm and t == '__m64':
        return '__m128'
      return t
    def translate_arg(t, a):
      if use_xmm and t == '__m64':
        return f'm128_to_m64({a})'
      return a

    arg_decl = ', '.join(f'{translate_type(v[1])} arg_{v[0]}' for v in enumerate(arg_tys)) or 'void'
    call_args = ', '.join(translate_arg(v[1], f'arg_{v[0]}') for v in enumerate(arg_tys))

    def create_fn(suffix, extraarg):
      if header:
        print(f'{ret_ty} {prefix}_{name}{suffix}({arg_decl});')
      else:
        print(f'{attr}{ret_ty} {prefix}_{name}{suffix}({arg_decl})')
        if use_xmm and convert_ret:
          print(f'{{ return ({ret_ty})m64_to_m128({name}({call_args}{extraarg})); }}')
        else:
          print(f'{{ return {name}({call_args}{extraarg}); }}')

    if imm_range:
      for i in range(imm_range[0], imm_range[1]+1):
        create_fn(f'_{i}', f', {i}')
    else:
      create_fn('', '')
    if condition:
      print('#endif')

  do_mmx(fn)


# Create a C file that tests an "orig" set of wrappers against a "new"
# set of wrappers.
def define_tests(use_xmm=False):
  def fn(name, ret_ty, arg_tys, imm_range=None, target=None, condition=None):
    if condition:
      print(f'#if {condition}')
    arg_decl = ', '.join(f'{v[1]} arg_{v[0]}' for v in enumerate(arg_tys)) or 'void'
    print(f' // {ret_ty} {name}({arg_decl});')

    if imm_range:
      for i in range(imm_range[0], imm_range[1]+1):
        fn(name + f'_{i}', ret_ty, arg_tys[:-1], target=target)
      return

    convert_pre = convert_post = ''
    if use_xmm and ret_ty == '__m64':
      convert_pre = 'm128_to_m64('
      convert_post = ')'

    args=[]
    loops=[]
    printf_fmts = []
    printf_args = []
    for arg_ty in arg_tys:
      v=len(loops)
      if arg_ty in ('char', 'short'):
        loops.append(f' for(int l{v} = 0; l{v} < arraysize(short_vals); ++l{v}) {{')
        args.append(f'({arg_ty})short_vals[l{v}]')
        printf_fmts.append('%016x')
        printf_args.append(f'short_vals[l{v}]')
      elif arg_ty in ('int', 'long long'):
        loops.append(f' for(int l{v} = 0; l{v} < arraysize(mmx_vals); ++l{v}) {{')
        args.append(f'({arg_ty})mmx_vals[l{v}]')
        printf_fmts.append('%016llx')
        printf_args.append(f'mmx_vals[l{v}]')
      elif arg_ty == '__m64':
        loops.append(f' for(int l{v} = 0; l{v} < arraysize(mmx_vals); ++l{v}) {{')
        if use_xmm:
          loops.append(f' for(int l{v+1} = 0; l{v+1} < arraysize(padding_mmx_vals); ++l{v+1}) {{')
          args.append(f'(__m128)(__m128i){{mmx_vals[l{v}], padding_mmx_vals[l{v+1}]}}')
          printf_fmts.append('(__m128i){%016llx, %016llx}')
          printf_args.append(f'mmx_vals[l{v}], padding_mmx_vals[l{v+1}]')
        else:
          args.append(f'({arg_ty})mmx_vals[l{v}]')
          printf_fmts.append('%016llx')
          printf_args.append(f'mmx_vals[l{v}]')
      elif arg_ty in ('__m128', '__m128i', '__m128d'):
        loops.append(f' for(int l{v} = 0; l{v} < arraysize(mmx_vals); ++l{v}) {{')
        loops.append(f' for(int l{v+1} = 0; l{v+1} < arraysize(mmx_vals); ++l{v+1}) {{')
        args.append(f'({arg_ty})(__m128i){{mmx_vals[l{v}], mmx_vals[l{v+1}]}}')
        printf_fmts.append('(__m128i){%016llx, %016llx}')
        printf_args.append(f'mmx_vals[l{v}], mmx_vals[l{v+1}]')
      elif arg_ty == 'const __m64 *':
        loops.append(f' for(int l{v} = 0; l{v} < arraysize(mmx_vals); ++l{v}) {{\n' +
                     f'  mem.m64 = (__m64)mmx_vals[l{v}];')
        args.append(f'&mem.m64')
        printf_fmts.append('&mem.m64 /* %016llx */')
        printf_args.append(f'(long long)mem.m64')
      else:
        print(' //   -> UNSUPPORTED')
        return

    printf_fmt_str = '"' + ', '.join(printf_fmts) + '"'
    if printf_args:
      printf_arg_str = ', ' + ','.join(printf_args)
    else:
      printf_arg_str = ''

    print('\n'.join(loops))
    print(f'''
  clear_exc_flags();
  {ret_ty} orig_res = {convert_pre}orig_{name}({", ".join(args)}){convert_post};
  int orig_exc = get_exc_flags();
  clear_exc_flags();
  {ret_ty} new_res = {convert_pre}new_{name}({", ".join(args)}){convert_post};
  int new_exc = get_exc_flags();
  check_mismatch("{name}", orig_exc, new_exc, &orig_res, &new_res, sizeof(orig_res), {printf_fmt_str}{printf_arg_str});
''')
    print(' }\n' * len(loops))
    print()
    if condition:
      print('#endif')

  do_mmx(fn)


parser = argparse.ArgumentParser(description='Generate mmx test code.')
parser.add_argument('--kind', choices=['wrapper', 'wrapper_h', 'test'])
parser.add_argument('--wrapper-prefix', default='orig')
parser.add_argument('--use-xmm', action='store_true')

args = parser.parse_args()
if args.kind == 'wrapper':
  define_wrappers(args.wrapper_prefix, use_xmm=args.use_xmm, header=False)
elif args.kind == 'wrapper_h':
  define_wrappers(args.wrapper_prefix, use_xmm=args.use_xmm, header=True)
elif args.kind == 'test':
  define_tests(use_xmm=args.use_xmm)
