; RUN: opt -dse -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%class.basic_string = type { %"class.__gnu_cxx::__versa_string" }
%"class.__gnu_cxx::__versa_string" = type { %"class.__gnu_cxx::__sso_string_base" }
%"class.__gnu_cxx::__sso_string_base" = type { %"struct.__gnu_cxx::__vstring_utility<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider", i64, %union.anon }
%"struct.__gnu_cxx::__vstring_utility<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" = type { ptr }
%union.anon = type { i64, [8 x i8] }

; Function Attrs: nounwind
declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1) #0

; Function Attrs: noinline nounwind readonly uwtable
declare zeroext i1 @callee_takes_string(ptr nonnull) #1 align 2

; Function Attrs: nounwind uwtable
define weak_odr zeroext i1 @test() #2 align 2 {

; CHECK-LABEL: @test

bb:
  %tmp = alloca %class.basic_string, align 8
  %tmp1 = alloca %class.basic_string, align 8
  %tmp3 = getelementptr inbounds %class.basic_string, ptr %tmp, i64 0, i32 0, i32 0, i32 2
  %tmp6 = getelementptr inbounds %class.basic_string, ptr %tmp, i64 0, i32 0, i32 0, i32 1
  %tmp7 = getelementptr inbounds i8, ptr %tmp3, i64 1
  %tmp9 = bitcast i64 0 to i64
  %tmp10 = getelementptr inbounds %class.basic_string, ptr %tmp1, i64 0, i32 0, i32 0, i32 2
  %tmp13 = getelementptr inbounds %class.basic_string, ptr %tmp1, i64 0, i32 0, i32 0, i32 1
  %tmp14 = getelementptr inbounds i8, ptr %tmp10, i64 1
  br label %_ZN12basic_stringIcSt11char_traitsIcESaIcEEC2EPKcRKS2_.exit

_ZN12basic_stringIcSt11char_traitsIcESaIcEEC2EPKcRKS2_.exit: ; preds = %bb
  store ptr %tmp3, ptr %tmp, align 8
  store i8 62, ptr %tmp3, align 8
  store i64 1, ptr %tmp6, align 8
  store i8 0, ptr %tmp7, align 1
  %tmp16 = call zeroext i1 @callee_takes_string(ptr nonnull %tmp)
  br label %_ZN9__gnu_cxx17__sso_string_baseIcSt11char_traitsIcESaIcEED2Ev.exit3

_ZN9__gnu_cxx17__sso_string_baseIcSt11char_traitsIcESaIcEED2Ev.exit3: ; preds = %_ZN12basic_stringIcSt11char_traitsIcESaIcEEC2EPKcRKS2_.exit

; CHECK: _ZN9__gnu_cxx17__sso_string_baseIcSt11char_traitsIcESaIcEED2Ev.exit3:

; The following can be read through the call %tmp17:
  store ptr %tmp10, ptr %tmp1, align 8
  store i8 125, ptr %tmp10, align 8
  store i64 1, ptr %tmp13, align 8
  store i8 0, ptr %tmp14, align 1

; CHECK: store ptr %tmp10, ptr %tmp1, align 8
; CHECK: store i8 125, ptr %tmp10, align 8
; CHECK: store i64 1, ptr %tmp13, align 8
; CHECK: store i8 0, ptr %tmp14, align 1

  %tmp17 = call zeroext i1 @callee_takes_string(ptr nonnull %tmp1)
  call void @llvm.memset.p0.i64(ptr align 8 %tmp10, i8 -51, i64 16, i1 false) #0
  call void @llvm.memset.p0.i64(ptr align 8 %tmp1, i8 -51, i64 32, i1 false) #0
  call void @llvm.memset.p0.i64(ptr align 8 %tmp3, i8 -51, i64 16, i1 false) #0
  call void @llvm.memset.p0.i64(ptr align 8 %tmp, i8 -51, i64 32, i1 false) #0
  ret i1 %tmp17
}

attributes #0 = { nounwind }
attributes #1 = { noinline nounwind readonly uwtable }
attributes #2 = { nounwind uwtable }

