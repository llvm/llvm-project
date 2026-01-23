; RUN: opt < %s -passes=pgo-instr-gen -enable-vtable-value-profiling -S 2>&1 | FileCheck %s --check-prefix=GEN --implicit-check-not="VTable value profiling is presently not supported"
; RUN: opt < %s -passes=pgo-instr-gen,instrprof -enable-vtable-value-profiling -S 2>&1 | FileCheck %s --check-prefix=LOWER --implicit-check-not="VTable value profiling is presently not supported"

source_filename = "vtable_local.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; The test IR is generated based on the following C++ program.
; Base1 has external linkage and Base2 has local linkage.
; class Derived uses multiple inheritance so its virtual table
; global variable contains two vtables. func1 is loaded from
; the vtable compatible with class Base1, and func2 is loaded
; from the vtable compatible with class Base2.

; class Base1 {
; public:
;   virtual int func1(int a) ;
; };
;
; namespace {
; class Base2 {
; public:
;   __attribute__((noinline)) virtual int func2(int a) {
;     return a;
;   }
; };
; }

; class Derived : public Base1, public Base2 {
; public:
;   Derived(int c) : v(c) {}
; private:
;   int v;
; };
;
; Derived* createType();

; int func(int a) {
;   Derived* d = createType();
;   return d->func2(a) + d->func1(a);
; }

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZTV7Derived = constant { [3 x ptr], [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN5Base15func1Ei], [3 x ptr] [ptr inttoptr (i64 -8 to ptr), ptr null, ptr @_ZN12_GLOBAL__N_15Base25func2Ei] }, !type !0, !type !3, !type !6, !type !8, !type !10
@_ZTV5Base1 = available_externally constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN5Base15func1Ei] }, !type !0
@_ZTVN12_GLOBAL__N_15Base2E = internal constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN12_GLOBAL__N_15Base25func2Ei] }, !type !11, !type !8; !vcall_visibility !12
@llvm.compiler.used = appending global [1 x ptr] [ptr @_ZTV5Base1], section "llvm.metadata"

; GEN: __llvm_profile_raw_version = comdat any
; GEN: __llvm_profile_raw_version = hidden constant i64 72057594037927946, comdat
; GEN: __profn__Z4funci = private constant [8 x i8] c"_Z4funci"

; LOWER: $__profvt__ZTV7Derived = comdat nodeduplicate
; LOWER: $"__profvt_vtable_local.ll;_ZTVN12_GLOBAL__N_15Base2E" = comdat nodeduplicate
; LOWER: @__profvt__ZTV7Derived = global { i64, ptr, i32 } { i64 -4576307468236080025, ptr @_ZTV7Derived, i32 48 }, section "__llvm_prf_vtab", comdat, align 8
; LOWER: @"__profvt_vtable_local.ll;_ZTVN12_GLOBAL__N_15Base2E" = internal global { i64, ptr, i32 } { i64 1419990121885302679, ptr @_ZTVN12_GLOBAL__N_15Base2E, i32 24 }, section "__llvm_prf_vtab", comdat, align 8
; LOWER: @__llvm_prf_vnm = private constant {{.*}}, section "__llvm_prf_vns", align 1
; LOWER: @llvm.used = appending global [5 x ptr] [ptr @__profvt__ZTV7Derived, ptr @"__profvt_vtable_local.ll;_ZTVN12_GLOBAL__N_15Base2E", ptr @__llvm_prf_vnodes, ptr @__llvm_prf_nm, ptr @__llvm_prf_vnm], section "llvm.metadata"

define i32 @_Z4funci(i32 %a) {
entry:
  %call = call ptr @_Z10createTypev()
  %add.ptr = getelementptr inbounds i8, ptr %call, i64 8
  %vtable = load ptr, ptr %add.ptr
; GEN: [[P1:%[0-9]+]] = ptrtoint ptr %vtable to i64
; GEN: call void @llvm.instrprof.value.profile(ptr @__profn__Z4funci, i64 [[CFGHash:[0-9]+]], i64 [[P1]], i32 2, i32 0)
; LOWER: [[P1:%[0-9]+]] = ptrtoint ptr %vtable to i64
; LOWER: call void @__llvm_profile_instrument_target(i64 [[P1]], ptr @__profd__Z4funci, i32 2)
  %vfunc1 = load ptr, ptr %vtable
  %call1 = call i32 %vfunc1(ptr %add.ptr, i32 %a)
  %vtable2 = load ptr, ptr %call
; GEN: [[P2:%[0-9]+]] = ptrtoint ptr %vtable2 to i64
; GEN: call void @llvm.instrprof.value.profile(ptr @__profn__Z4funci, i64 [[CFGHash]], i64 [[P2]], i32 2, i32 1)
; LOWER: [[P2:%[0-9]+]] = ptrtoint ptr %vtable2 to i64
; LOWER: call void @__llvm_profile_instrument_target(i64 [[P2]], ptr @__profd__Z4funci, i32 3)
  %vfunc2 = load ptr, ptr %vtable2
  %call4 = call i32 %vfunc2(ptr %call, i32 %a)
  %add = add nsw i32 %call1, %call4
  ret i32 %add
}

declare ptr @_Z10createTypev()
declare i32 @_ZN12_GLOBAL__N_15Base25func2Ei(ptr %this, i32 %a)
declare i32 @_ZN5Base15func1Ei(ptr, i32)

!0 = !{i64 16, !"_ZTS5Base1"}
!3 = !{i64 16, !"_ZTS7Derived"}
!6 = !{i64 40, !7}
!7 = distinct !{}
!8 = !{i64 16, !9}
!9 = distinct !{}
!10 = !{i64 40, !9}
!11 = !{i64 16, !7}
