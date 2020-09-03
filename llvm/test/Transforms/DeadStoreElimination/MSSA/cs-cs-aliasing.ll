; RUN: opt -basic-aa -dse -enable-dse-memoryssa -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%class.basic_string = type { %"class.__gnu_cxx::__versa_string" }
%"class.__gnu_cxx::__versa_string" = type { %"class.__gnu_cxx::__sso_string_base" }
%"class.__gnu_cxx::__sso_string_base" = type { %"struct.__gnu_cxx::__vstring_utility<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider", i64, %union.anon }
%"struct.__gnu_cxx::__vstring_utility<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" = type { i8* }
%union.anon = type { i64, [8 x i8] }

; Function Attrs: nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) #0

; Function Attrs: noinline nounwind readonly uwtable
declare zeroext i1 @callee_takes_string(%class.basic_string* nonnull) #1 align 2

; Function Attrs: nounwind uwtable
define weak_odr zeroext i1 @test() #2 align 2 {

; CHECK-LABEL: @test

bb:
  %tmp = alloca %class.basic_string, align 8
  %tmp1 = alloca %class.basic_string, align 8
  %tmp3 = getelementptr inbounds %class.basic_string, %class.basic_string* %tmp, i64 0, i32 0, i32 0, i32 2
  %tmp4 = bitcast %union.anon* %tmp3 to i8*
  %tmp5 = getelementptr inbounds %class.basic_string, %class.basic_string* %tmp, i64 0, i32 0, i32 0, i32 0, i32 0
  %tmp6 = getelementptr inbounds %class.basic_string, %class.basic_string* %tmp, i64 0, i32 0, i32 0, i32 1
  %tmp7 = getelementptr inbounds i8, i8* %tmp4, i64 1
  %tmp8 = bitcast %class.basic_string* %tmp to i8*
  %tmp9 = bitcast i64 0 to i64
  %tmp10 = getelementptr inbounds %class.basic_string, %class.basic_string* %tmp1, i64 0, i32 0, i32 0, i32 2
  %tmp11 = bitcast %union.anon* %tmp10 to i8*
  %tmp12 = getelementptr inbounds %class.basic_string, %class.basic_string* %tmp1, i64 0, i32 0, i32 0, i32 0, i32 0
  %tmp13 = getelementptr inbounds %class.basic_string, %class.basic_string* %tmp1, i64 0, i32 0, i32 0, i32 1
  %tmp14 = getelementptr inbounds i8, i8* %tmp11, i64 1
  %tmp15 = bitcast %class.basic_string* %tmp1 to i8*
  br label %_ZN12basic_stringIcSt11char_traitsIcESaIcEEC2EPKcRKS2_.exit

_ZN12basic_stringIcSt11char_traitsIcESaIcEEC2EPKcRKS2_.exit: ; preds = %bb
  store i8* %tmp4, i8** %tmp5, align 8
  store i8 62, i8* %tmp4, align 8
  store i64 1, i64* %tmp6, align 8
  store i8 0, i8* %tmp7, align 1
  %tmp16 = call zeroext i1 @callee_takes_string(%class.basic_string* nonnull %tmp)
  br label %_ZN9__gnu_cxx17__sso_string_baseIcSt11char_traitsIcESaIcEED2Ev.exit3

_ZN9__gnu_cxx17__sso_string_baseIcSt11char_traitsIcESaIcEED2Ev.exit3: ; preds = %_ZN12basic_stringIcSt11char_traitsIcESaIcEEC2EPKcRKS2_.exit

; CHECK: _ZN9__gnu_cxx17__sso_string_baseIcSt11char_traitsIcESaIcEED2Ev.exit3:

; The following can be read through the call %tmp17:
  store i8* %tmp11, i8** %tmp12, align 8
  store i8 125, i8* %tmp11, align 8
  store i64 1, i64* %tmp13, align 8
  store i8 0, i8* %tmp14, align 1

; CHECK: store i8* %tmp11, i8** %tmp12, align 8
; CHECK: store i8 125, i8* %tmp11, align 8
; CHECK: store i64 1, i64* %tmp13, align 8
; CHECK: store i8 0, i8* %tmp14, align 1

  %tmp17 = call zeroext i1 @callee_takes_string(%class.basic_string* nonnull %tmp1)
  call void @llvm.memset.p0i8.i64(i8* align 8 %tmp11, i8 -51, i64 16, i1 false) #0
  call void @llvm.memset.p0i8.i64(i8* align 8 %tmp15, i8 -51, i64 32, i1 false) #0
  call void @llvm.memset.p0i8.i64(i8* align 8 %tmp4, i8 -51, i64 16, i1 false) #0
  call void @llvm.memset.p0i8.i64(i8* align 8 %tmp8, i8 -51, i64 32, i1 false) #0
  ret i1 %tmp17
}

attributes #0 = { nounwind }
attributes #1 = { noinline nounwind readonly uwtable }
attributes #2 = { nounwind uwtable }

