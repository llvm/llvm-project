; RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -march=amdgcn -mcpu=gfx900 -verify-machineinstrs | FileCheck %s

; CHECK: s_add_i32 s0, s0, s1
; CHECK: s_add_i32 s1, s0, s2
; CHECK: s_add_i32 s2, s1, s3
; CHECK: s_add_i32 s3, s2, s4
; CHECK: s_add_i32 s4, s3, s5
; CHECK: s_add_i32 s5, s4, s6
; CHECK: s_add_i32 s6, s5, s7
; CHECK: s_add_i32 s7, s6, s8
; CHECK: s_add_i32 s8, s7, s9
; CHECK: s_add_i32 s9, s8, s10
; CHECK: s_add_i32 s10, s9, s11
; CHECK: s_add_i32 s11, s10, s12
; CHECK: s_add_i32 s12, s11, s13
; CHECK: s_add_i32 s13, s12, s14
; CHECK: s_add_i32 s14, s13, s15
; CHECK: s_add_i32 s15, s14, s16
; CHECK: s_add_i32 s16, s15, s17
; CHECK: s_add_i32 s17, s16, s18
; CHECK: s_add_i32 s18, s17, s19
; CHECK: s_add_i32 s19, s18, s20
; CHECK: s_add_i32 s20, s19, s21
; CHECK: s_add_i32 s21, s20, s22
; CHECK: s_add_i32 s22, s21, s23
; CHECK: s_add_i32 s23, s22, s24
; CHECK: s_add_i32 s24, s23, s25
; CHECK: s_add_i32 s25, s24, s26
; CHECK: s_add_i32 s26, s25, s27
; CHECK: s_add_i32 s27, s26, s28
; CHECK: s_add_i32 s28, s27, s29
define amdgpu_gs { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } @_amdgpu_gs_sgpr_limit_i32 (i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg) {
.entry:
  %30 = add i32 %0, %1
  %31 =  add i32 %30, %2
  %32 =  add i32 %31, %3
  %33 =  add i32 %32, %4
  %34 =  add i32 %33, %5
  %35 =  add i32 %34, %6
  %36 =  add i32 %35, %7
  %37 =  add i32 %36, %8
  %38 =  add i32 %37, %9
  %39 =  add i32 %38, %10
  %40 =  add i32 %39, %11
  %41 =  add i32 %40, %12
  %42 =  add i32 %41, %13
  %43 =  add i32 %42, %14
  %44 =  add i32 %43, %15
  %45 =  add i32 %44, %16
  %46 =  add i32 %45, %17
  %47 =  add i32 %46, %18
  %48 =  add i32 %47, %19
  %49 =  add i32 %48, %20
  %50 =  add i32 %49, %21
  %51 =  add i32 %50, %22
  %52 =  add i32 %51, %23
  %53 =  add i32 %52, %24
  %54 =  add i32 %53, %25
  %55 =  add i32 %54, %26
  %56 =  add i32 %55, %27
  %57 =  add i32 %56, %28
  %58 =  add i32 %57, %29
  %59 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } undef, i32 %30, 0
  %60 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %59, i32 %31, 1
  %61 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %60, i32 %32, 2
  %62 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %61, i32 %33, 3
  %63 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %62, i32 %34, 4
  %64 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %63, i32 %35, 5
  %65 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %64, i32 %36, 6
  %66 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %65, i32 %37, 7
  %67 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %66, i32 %38, 8
  %68 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %67, i32 %39, 9
  %69 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %68, i32 %40, 10
  %70 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %69, i32 %41, 11
  %71 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %70, i32 %42, 12
  %72 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %71, i32 %43, 13
  %73 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %72, i32 %44, 14
  %74 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %73, i32 %45, 15
  %75 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %74, i32 %46, 16
  %76 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %75, i32 %47, 17
  %77 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %76, i32 %48, 18
  %78 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %77, i32 %49, 19
  %79 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %78, i32 %50, 20
  %80 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %79, i32 %51, 21
  %81 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %80, i32 %52, 22
  %82 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %81, i32 %53, 23
  %83 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %82, i32 %54, 24
  %84 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %83, i32 %55, 25
  %85 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %84, i32 %56, 26
  %86 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %85, i32 %57, 27
  %87 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %86, i32 %58, 28
  ret { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %87
}

; CHECK: s_xor_b64 s[0:1], s[0:1], s[2:3]
; CHECK: s_xor_b64 s[0:1], s[0:1], s[4:5]
; CHECK: s_xor_b64 s[0:1], s[0:1], s[6:7]
; CHECK: s_xor_b64 s[0:1], s[0:1], s[8:9]
; CHECK: s_xor_b64 s[0:1], s[0:1], s[10:11]
; CHECK: s_xor_b64 s[0:1], s[0:1], s[12:13]
; CHECK: s_xor_b64 s[0:1], s[0:1], s[14:15]
; CHECK: s_xor_b64 s[0:1], s[0:1], s[16:17]
; CHECK: s_xor_b64 s[0:1], s[0:1], s[18:19]
; CHECK: s_xor_b64 s[0:1], s[0:1], s[20:21]
; CHECK: s_xor_b64 s[0:1], s[0:1], s[22:23]
; CHECK: s_xor_b64 s[0:1], s[0:1], s[24:25]
; CHECK: s_xor_b64 s[0:1], s[0:1], s[26:27]
; CHECK: s_xor_b64 s[0:1], s[0:1], s[28:29]
; CHECK: s_xor_b64 s[0:1], s[0:1], s[30:31]
; CHECK: s_xor_b64 s[0:1], s[0:1], s[32:33]
; CHECK: s_xor_b64 s[0:1], s[0:1], s[34:35]
define amdgpu_gs void @_amdgpu_gs_sgpr_limit_i64(i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, <4 x i32> inreg %addr) {
.entry:
  %19 = xor i64 %0, %1
  %20 =  xor i64 %19, %2
  %21 =  xor i64 %20, %3
  %22 =  xor i64 %21, %4
  %23 =  xor i64 %22, %5
  %24 =  xor i64 %23, %6
  %25 =  xor i64 %24, %7
  %26 =  xor i64 %25, %8
  %27 =  xor i64 %26, %9
  %28 =  xor i64 %27, %10
  %29 =  xor i64 %28, %11
  %30 =  xor i64 %29, %12
  %31 =  xor i64 %30, %13
  %32 =  xor i64 %31, %14
  %33 =  xor i64 %32, %15
  %34 =  xor i64 %33, %16
  %35 =  xor i64 %34, %17
  %36 = bitcast i64 %35 to <2 x i32>
  call void @llvm.amdgcn.raw.buffer.store.v2i32(<2 x i32> %36, <4 x i32> %addr, i32 4, i32 0, i32 0)
  ret void
}

declare void @llvm.amdgcn.raw.buffer.store.v2i32(<2 x i32>, <4 x i32>, i32, i32, i32)
