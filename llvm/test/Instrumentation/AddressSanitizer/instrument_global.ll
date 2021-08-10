; RUN: opt < %s -asan -asan-module -enable-new-pm=0 -asan-globals-live-support=1 -S | FileCheck %s
; RUN: opt < %s -passes='asan-pipeline' -asan-globals-live-support=1 -S | FileCheck %s
; RUN: opt < %s -asan -asan-module -enable-new-pm=0 -asan-globals-live-support=1 -asan-mapping-scale=5 -S | FileCheck %s
; RUN: opt < %s -passes='asan-pipeline' -asan-globals-live-support=1 -asan-mapping-scale=5 -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
@xxx = global i32 0, align 4

; If a global is present, __asan_[un]register_globals should be called from
; module ctor/dtor

; CHECK: @___asan_gen_ = private constant [8 x i8] c"<stdin>\00", align 1
; CHECK: @llvm.used = appending global [2 x i8*] [i8* bitcast (void ()* @asan.module_ctor to i8*), i8* bitcast (void ()* @asan.module_dtor to i8*)], section "llvm.metadata"
; CHECK: @llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 1, void ()* @asan.module_ctor, i8* bitcast (void ()* @asan.module_ctor to i8*) }]
; CHECK: @llvm.global_dtors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 1, void ()* @asan.module_dtor, i8* bitcast (void ()* @asan.module_dtor to i8*) }]

; Test that we don't instrument global arrays with static initializer
; indexed with constants in-bounds. But instrument all other cases.

@GlobSt = global [10 x i32] zeroinitializer, align 16  ; static initializer
@GlobDy = global [10 x i32] zeroinitializer, align 16  ; dynamic initializer
@GlobEx = external global [10 x i32] , align 16        ; extern initializer

; GlobSt is declared here, and has static initializer -- ok to optimize.
define i32 @AccessGlobSt_0_2() sanitize_address {
entry:
    %0 = load i32, i32* getelementptr inbounds ([10 x i32], [10 x i32]* @GlobSt, i64 0, i64 2), align 8
    ret i32 %0
; CHECK-LABEL: define i32 @AccessGlobSt_0_2
; CHECK-NOT: __asan_report
; CHECK: ret i32 %0
}

; GlobSt is accessed out of bounds -- can't optimize
define i32 @AccessGlobSt_0_12() sanitize_address {
entry:
    %0 = load i32, i32* getelementptr inbounds ([10 x i32], [10 x i32]* @GlobSt, i64 0, i64 12), align 8
    ret i32 %0
; CHECK-LABEL: define i32 @AccessGlobSt_0_12
; CHECK: __asan_report
; CHECK: ret i32
}

; GlobSt is accessed with Gep that has non-0 first index -- can't optimize.
define i32 @AccessGlobSt_1_2() sanitize_address {
entry:
    %0 = load i32, i32* getelementptr inbounds ([10 x i32], [10 x i32]* @GlobSt, i64 1, i64 2), align 8
    ret i32 %0
; CHECK-LABEL: define i32 @AccessGlobSt_1_2
; CHECK: __asan_report
; CHECK: ret i32
}

; GlobDy is declared with dynamic initializer -- can't optimize.
define i32 @AccessGlobDy_0_2() sanitize_address {
entry:
    %0 = load i32, i32* getelementptr inbounds ([10 x i32], [10 x i32]* @GlobDy, i64 0, i64 2), align 8
    ret i32 %0
; CHECK-LABEL: define i32 @AccessGlobDy_0_2
; CHECK: __asan_report
; CHECK: ret i32
}

; GlobEx is an external global -- can't optimize.
define i32 @AccessGlobEx_0_2() sanitize_address {
entry:
    %0 = load i32, i32* getelementptr inbounds ([10 x i32], [10 x i32]* @GlobEx, i64 0, i64 2), align 8
    ret i32 %0
; CHECK-LABEL: define i32 @AccessGlobEx_0_2
; CHECK: __asan_report
; CHECK: ret i32
}


!llvm.asan.globals = !{!0}
!0 = !{[10 x i32]* @GlobDy, null, null, i1 true, i1 false}

; CHECK-LABEL: define internal void @asan.module_ctor
; CHECK-NOT: ret
; CHECK: call void @__asan_register_elf_globals
; CHECK: ret

; CHECK-LABEL: define internal void @asan.module_dtor
; CHECK-NOT: ret
; CHECK: call void @__asan_unregister_elf_globals
; CHECK: ret
