; REQUIRES: asserts
; RUN: opt -passes=mergefunc -stats -disable-output < %s 2>&1 | grep "functions merged"

; This test is checks whether we can merge
;   vector<intptr_t>::push_back(0)
; and
;   vector<void *>::push_back(0)
; .

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%0 = type { i32, void ()*, i8* }
%1 = type { i64, i1 }
%"class.std::vector" = type { [24 x i8] }

@vi = global %"class.std::vector" zeroinitializer, align 8
@__dso_handle = external unnamed_addr global i8*
@vp = global %"class.std::vector" zeroinitializer, align 8
@llvm.global_ctors = appending global [1 x %0] [%0 { i32 65535, void ()* @_GLOBAL__I_a, i8* null }]

define linkonce_odr void @_ZNSt6vectorIlSaIlEED1Ev(%"class.std::vector"* nocapture %this) unnamed_addr align 2 {
entry:
  %tmp2.i.i = bitcast %"class.std::vector"* %this to i64**
  %tmp3.i.i = load i64*, i64** %tmp2.i.i, align 8
  %tobool.i.i.i = icmp eq i64* %tmp3.i.i, null
  br i1 %tobool.i.i.i, label %_ZNSt6vectorIlSaIlEED2Ev.exit, label %if.then.i.i.i

if.then.i.i.i:                                    ; preds = %entry
  %0 = bitcast i64* %tmp3.i.i to i8*
  tail call void @_ZdlPv(i8* %0) nounwind
  ret void

_ZNSt6vectorIlSaIlEED2Ev.exit:                    ; preds = %entry
  ret void
}

declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*)

define linkonce_odr void @_ZNSt6vectorIPvSaIS0_EED1Ev(%"class.std::vector"* nocapture %this) unnamed_addr align 2 {
entry:
  %tmp2.i.i = bitcast %"class.std::vector"* %this to i8***
  %tmp3.i.i = load i8**, i8*** %tmp2.i.i, align 8
  %tobool.i.i.i = icmp eq i8** %tmp3.i.i, null
  br i1 %tobool.i.i.i, label %_ZNSt6vectorIPvSaIS0_EED2Ev.exit, label %if.then.i.i.i

if.then.i.i.i:                                    ; preds = %entry
  %0 = bitcast i8** %tmp3.i.i to i8*
  tail call void @_ZdlPv(i8* %0) nounwind
  ret void

_ZNSt6vectorIPvSaIS0_EED2Ev.exit:                 ; preds = %entry
  ret void
}

declare void @_Z1fv()

declare void @_ZNSt6vectorIPvSaIS0_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS0_S2_EERKS0_(%"class.std::vector"* nocapture %this, i8** %__position.coerce, i8** nocapture %__x) align 2

declare void @_ZdlPv(i8*) nounwind

declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) nounwind

declare void @_ZSt17__throw_bad_allocv() noreturn

declare noalias i8* @_Znwm(i64)

declare void @_ZNSt6vectorIlSaIlEE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPlS1_EERKl(%"class.std::vector"* nocapture %this, i64* %__position.coerce, i64* nocapture %__x) align 2

declare void @_GLOBAL__I_a()

declare %1 @llvm.uadd.with.overflow.i64(i64, i64) nounwind readnone
