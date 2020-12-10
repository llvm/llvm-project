; Test that global metadata is placed in a separate section on Mach-O platforms,
; allowing dead stripping to be performed, and that the appropriate runtime
; routines are invoked.

; RUN: opt < %s -asan -asan-module -enable-new-pm=0 -asan-use-private-alias=0 -asan-globals-live-support=1 -S | FileCheck %s --check-prefixes=CHECK,NOALIAS
; RUN: opt < %s -passes='asan-pipeline'             -asan-use-private-alias=0 -asan-globals-live-support=1 -S | FileCheck %s --check-prefixes=CHECK,NOALIAS
; RUN: opt < %s -asan -asan-module -enable-new-pm=0 -asan-use-private-alias=1 -asan-globals-live-support=1 -S | FileCheck %s --check-prefixes=CHECK,ALIAS
; RUN: opt < %s -passes='asan-pipeline'             -asan-use-private-alias=1 -asan-globals-live-support=1 -S | FileCheck %s --check-prefixes=CHECK,ALIAS

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

@global = global [1 x i32] zeroinitializer, align 4

!llvm.asan.globals = !{!0}

!0 = !{[1 x i32]* @global, !1, !"global", i1 false, i1 false}
!1 = !{!"test-globals.c", i32 1, i32 5}


; Find the metadata for @global:
; NOALIAS: [[METADATA:@.+]] = internal global {{.*}} @global {{.*}} section "__DATA,__asan_globals,regular"
; ALIAS: [[METADATA:@.+]] = internal global {{.*}} @0 {{.*}} section "__DATA,__asan_globals,regular"

; Find the liveness binder for @global and its metadata:
; NOALIAS: @__asan_binder_global = internal global {{.*}} @global {{.*}} [[METADATA]] {{.*}} section "__DATA,__asan_liveness,regular,live_support"
; ALIAS: @__asan_binder_global = internal global {{.*}} @0 {{.*}} [[METADATA]] {{.*}} section "__DATA,__asan_liveness,regular,live_support"

; The binder has to be inserted to llvm.compiler.used to avoid being stripped
; during LTO.
; CHECK: @llvm.compiler.used {{.*}} @__asan_binder_global {{.*}} section "llvm.metadata"

; Test that there is the flag global variable:
; CHECK: @___asan_globals_registered = common hidden global i64 0

; ALIAS: @0 = private alias {{.*}} @global

; Test that __asan_register_image_globals is invoked from the constructor:
; CHECK-LABEL: define internal void @asan.module_ctor
; CHECK-NOT: ret
; CHECK: call void @__asan_register_image_globals(i64 ptrtoint (i64* @___asan_globals_registered to i64))
; CHECK: ret

; Test that __asan_unregister_image_globals is invoked from the destructor:
; CHECK-LABEL: define internal void @asan.module_dtor
; CHECK-NOT: ret
; CHECK: call void @__asan_unregister_image_globals(i64 ptrtoint (i64* @___asan_globals_registered to i64))
; CHECK: ret
