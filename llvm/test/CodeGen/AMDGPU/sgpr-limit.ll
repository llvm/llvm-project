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
; CHECK: s_add_i32 s29, s28, s30
; CHECK: s_add_i32 s30, s29, s31
; CHECK: s_add_i32 s31, s30, s32
; CHECK: s_add_i32 s32, s31, s33
; CHECK: s_add_i32 s33, s32, s34
; CHECK: s_add_i32 s34, s33, s35
; CHECK: s_add_i32 s35, s34, s36
; CHECK: s_add_i32 s36, s35, s37
; CHECK: s_add_i32 s37, s36, s38
; CHECK: s_add_i32 s38, s37, s39
; CHECK: s_add_i32 s39, s38, s40
; CHECK: s_add_i32 s40, s39, s41
; CHECK: s_add_i32 s41, s40, s42
; CHECK: s_add_i32 s42, s41, s43
; CHECK: s_add_i32 s43, s42, s44
; CHECK: s_add_i32 s44, s43, s45
; CHECK: s_add_i32 s45, s44, s46
; CHECK: s_add_i32 s46, s45, s47
; CHECK: s_add_i32 s47, s46, s48
; CHECK: s_add_i32 s48, s47, s49
; CHECK: s_add_i32 s49, s48, s50
; CHECK: s_add_i32 s50, s49, s51
; CHECK: s_add_i32 s51, s50, s52
; CHECK: s_add_i32 s52, s51, s53
; CHECK: s_add_i32 s53, s52, s54
; CHECK: s_add_i32 s54, s53, s55
; CHECK: s_add_i32 s55, s54, s56
; CHECK: s_add_i32 s56, s55, s57
; CHECK: s_add_i32 s57, s56, s58
; CHECK: s_add_i32 s58, s57, s59
; CHECK: s_add_i32 s59, s58, s60
; CHECK: s_add_i32 s60, s59, s61
; CHECK: s_add_i32 s61, s60, s62
; CHECK: s_add_i32 s62, s61, s63
define amdgpu_gs  { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } @_amdgpu_gs_sgpr_limit_i32 (i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, <4 x i32> inreg) {
.entry:
  %65 = add i32 %0, %1
  %66 =  add i32 %65, %2
  %67 =  add i32 %66, %3
  %68 =  add i32 %67, %4
  %69 =  add i32 %68, %5
  %70 =  add i32 %69, %6
  %71 =  add i32 %70, %7
  %72 =  add i32 %71, %8
  %73 =  add i32 %72, %9
  %74 =  add i32 %73, %10
  %75 =  add i32 %74, %11
  %76 =  add i32 %75, %12
  %77 =  add i32 %76, %13
  %78 =  add i32 %77, %14
  %79 =  add i32 %78, %15
  %80 =  add i32 %79, %16
  %81 =  add i32 %80, %17
  %82 =  add i32 %81, %18
  %83 =  add i32 %82, %19
  %84 =  add i32 %83, %20
  %85 =  add i32 %84, %21
  %86 =  add i32 %85, %22
  %87 =  add i32 %86, %23
  %88 =  add i32 %87, %24
  %89 =  add i32 %88, %25
  %90 =  add i32 %89, %26
  %91 =  add i32 %90, %27
  %92 =  add i32 %91, %28
  %93 =  add i32 %92, %29
  %94 =  add i32 %93, %30
  %95 =  add i32 %94, %31
  %96 =  add i32 %95, %32
  %97 =  add i32 %96, %33
  %98 =  add i32 %97, %34
  %99 =  add i32 %98, %35
  %100 =  add i32 %99, %36
  %101 =  add i32 %100, %37
  %102 =  add i32 %101, %38
  %103 =  add i32 %102, %39
  %104 =  add i32 %103, %40
  %105 =  add i32 %104, %41
  %106 =  add i32 %105, %42
  %107 =  add i32 %106, %43
  %108 =  add i32 %107, %44
  %109 =  add i32 %108, %45
  %110 =  add i32 %109, %46
  %111 =  add i32 %110, %47
  %112 =  add i32 %111, %48
  %113 =  add i32 %112, %49
  %114 =  add i32 %113, %50
  %115 =  add i32 %114, %51
  %116 =  add i32 %115, %52
  %117 =  add i32 %116, %53
  %118 =  add i32 %117, %54
  %119 =  add i32 %118, %55
  %120 =  add i32 %119, %56
  %121 =  add i32 %120, %57
  %122 =  add i32 %121, %58
  %123 =  add i32 %122, %59
  %124 =  add i32 %123, %60
  %125 =  add i32 %124, %61
  %126 =  add i32 %125, %62
  %127 =  add i32 %126, %63
%128 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } undef, i32 %65, 0
%129 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %128, i32 %66, 1
%130 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %129, i32 %67, 2
%131 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %130, i32 %68, 3
%132 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %131, i32 %69, 4
%133 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %132, i32 %70, 5
%134 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %133, i32 %71, 6
%135 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %134, i32 %72, 7
%136 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %135, i32 %73, 8
%137 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %136, i32 %74, 9
%138 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %137, i32 %75, 10
%139 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %138, i32 %76, 11
%140 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %139, i32 %77, 12
%141 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %140, i32 %78, 13
%142 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %141, i32 %79, 14
%143 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %142, i32 %80, 15
%144 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %143, i32 %81, 16
%145 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %144, i32 %82, 17
%146 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %145, i32 %83, 18
%147 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %146, i32 %84, 19
%148 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %147, i32 %85, 20
%149 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %148, i32 %86, 21
%150 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %149, i32 %87, 22
%151 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %150, i32 %88, 23
%152 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %151, i32 %89, 24
%153 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %152, i32 %90, 25
%154 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %153, i32 %91, 26
%155 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %154, i32 %92, 27
%156 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %155, i32 %93, 28
%157 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %156, i32 %94, 29
%158 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %157, i32 %95, 30
%159 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %158, i32 %96, 31
%160 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %159, i32 %97, 32
%161 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %160, i32 %98, 33
%162 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %161, i32 %99, 34
%163 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %162, i32 %100, 35
%164 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %163, i32 %101, 36
%165 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %164, i32 %102, 37
%166 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %165, i32 %103, 38
%167 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %166, i32 %104, 39
%168 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %167, i32 %105, 40
%169 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %168, i32 %106, 41
%170 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %169, i32 %107, 42
%171 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %170, i32 %108, 43
%172 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %171, i32 %109, 44
%173 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %172, i32 %110, 45
%174 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %173, i32 %111, 46
%175 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %174, i32 %112, 47
%176 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %175, i32 %113, 48
%177 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %176, i32 %114, 49
%178 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %177, i32 %115, 50
%179 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %178, i32 %116, 51
%180 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %179, i32 %117, 52
%181 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %180, i32 %118, 53
%182 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %181, i32 %119, 54
%183 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %182, i32 %120, 55
%184 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %183, i32 %121, 56
%185 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %184, i32 %122, 57
%186 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %185, i32 %123, 58
%187 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %186, i32 %124, 59
%188 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %187, i32 %125, 60
%189 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %188, i32 %126, 61
%190 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %189, i32 %127, 62
  ret { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %190
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
; CHECK: s_xor_b64 s[0:1], s[0:1], s[36:37]
; CHECK: s_xor_b64 s[0:1], s[0:1], s[38:39]
; CHECK: s_xor_b64 s[0:1], s[0:1], s[40:41]
; CHECK: s_xor_b64 s[0:1], s[0:1], s[42:43]
; CHECK: s_xor_b64 s[0:1], s[0:1], s[44:45]
; CHECK: s_xor_b64 s[0:1], s[0:1], s[46:47]
; CHECK: s_xor_b64 s[0:1], s[0:1], s[48:49]
; CHECK: s_xor_b64 s[0:1], s[0:1], s[50:51]
; CHECK: s_xor_b64 s[0:1], s[0:1], s[52:53]
; CHECK: s_xor_b64 s[0:1], s[0:1], s[54:55]
; CHECK: s_xor_b64 s[0:1], s[0:1], s[56:57]
; CHECK: s_xor_b64 s[0:1], s[0:1], s[58:59]
define amdgpu_gs void @_amdgpu_gs_sgpr_limit_i64 (i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, i64 inreg, <4 x i32> inreg %addr) {
.entry:
  %31 = xor i64 %0, %1
  %32 = xor i64 %31, %2
  %33 = xor i64 %32, %3
  %34 = xor i64 %33, %4
  %35 = xor i64 %34, %5
  %36 = xor i64 %35, %6
  %37 = xor i64 %36, %7
  %38 = xor i64 %37, %8
  %39 = xor i64 %38, %9
  %40 = xor i64 %39, %10
  %41 = xor i64 %40, %11
  %42 = xor i64 %41, %12
  %43 = xor i64 %42, %13
  %44 = xor i64 %43, %14
  %45 = xor i64 %44, %15
  %46 = xor i64 %45, %16
  %47 = xor i64 %46, %17
  %48 = xor i64 %47, %18
  %49 = xor i64 %48, %19
  %50 = xor i64 %49, %20
  %51 = xor i64 %50, %21
  %52 = xor i64 %51, %22
  %53 = xor i64 %52, %23
  %54 = xor i64 %53, %24
  %55 = xor i64 %54, %25
  %56 = xor i64 %55, %26
  %57 = xor i64 %56, %27
  %58 = xor i64 %57, %28
  %59 = xor i64 %58, %29
  %60 = bitcast i64 %59 to <2 x i32>
  call void @llvm.amdgcn.raw.buffer.store.v2i32(<2 x i32> %60, <4 x i32> %addr, i32 4, i32 0, i32 0)
  ret void
}

declare void @llvm.amdgcn.raw.buffer.store.v2i32(<2 x i32>, <4 x i32>, i32, i32, i32)

