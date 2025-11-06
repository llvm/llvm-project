; RUN: opt < %s -passes=asan -asan-globals-live-support=1 -S | FileCheck %s
; RUN: opt < %s -passes=asan -asan-globals-live-support=1 -asan-mapping-scale=5 -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
@xxx = global i32 0, align 4

; If a global is present, __asan_[un]register_globals should be called from
; module ctor/dtor

; CHECK: @___asan_gen_module = private constant [8 x i8] c"<stdin>\00", align 1
; CHECK: @llvm.used = appending global [2 x ptr] [ptr @asan.module_ctor, ptr @asan.module_dtor], section "llvm.metadata"
; CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @asan.module_ctor, ptr @asan.module_ctor }]
; CHECK: @llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @asan.module_dtor, ptr @asan.module_dtor }]

; Test that we don't instrument global arrays with static initializer
; indexed with constants in-bounds. But instrument all other cases.

@GlobSt = global [10 x i32] zeroinitializer, align 16  ; static initializer
@GlobStAlignInBounds = global [10 x i8] zeroinitializer, align 16  ; static initializer
@GlobDy = global [10 x i32] zeroinitializer, align 16, sanitize_address_dyninit  ; dynamic initializer
@GlobEx = external global [10 x i32] , align 16        ; extern initializer

; GlobSt is declared here, and has static initializer -- ok to optimize.
define i32 @AccessGlobSt_0_2() sanitize_address {
entry:
    %0 = load i32, ptr getelementptr inbounds ([10 x i32], ptr @GlobSt, i64 0, i64 2), align 8
    ret i32 %0
; CHECK-LABEL: define i32 @AccessGlobSt_0_2
; CHECK-NOT: __asan_report
; CHECK: ret i32 %0
}

; GlobSt is accessed out of bounds -- can't optimize
define i32 @AccessGlobSt_0_12() sanitize_address {
entry:
    %0 = load i32, ptr getelementptr inbounds ([10 x i32], ptr @GlobSt, i64 0, i64 12), align 8
    ret i32 %0
; CHECK-LABEL: define i32 @AccessGlobSt_0_12
; CHECK: __asan_report
; CHECK: ret i32
}

; GlobSt is accessed with Gep that has non-0 first index -- can't optimize.
define i32 @AccessGlobSt_1_2() sanitize_address {
entry:
    %0 = load i32, ptr getelementptr inbounds ([10 x i32], ptr @GlobSt, i64 1, i64 2), align 8
    ret i32 %0
; CHECK-LABEL: define i32 @AccessGlobSt_1_2
; CHECK: __asan_report
; CHECK: ret i32
}

; GlobStAlignInBount is accessed with out of bounds index, but in bounds of allocated area (because of alignemnt)
define i8 @AccessGlobStAlignInBounds_0_11() sanitize_address {
entry:
    %0 = load i8, ptr getelementptr inbounds ([10 x i8], ptr @GlobStAlignInBounds, i64 0, i64 11), align 1
    ret i8 %0
; CHECK-LABEL: define i8 @AccessGlobStAlignInBounds_0_11
; CHECK: __asan_report
; CHECK: ret i8
}

; GlobStAlignInBount is accessed with in-bound index
define i8 @AccessGlobStAlignInBounds_0_9() sanitize_address {
entry:
    %0 = load i8, ptr getelementptr inbounds ([10 x i8], ptr @GlobStAlignInBounds, i64 0, i64 9), align 1
    ret i8 %0
; CHECK-LABEL: define i8 @AccessGlobStAlignInBounds_0_9
; CHECK-NOT: __asan_report
; CHECK: ret i8
}

; GlobDy is declared with dynamic initializer -- can't optimize.
define i32 @AccessGlobDy_0_2() sanitize_address {
entry:
    %0 = load i32, ptr getelementptr inbounds ([10 x i32], ptr @GlobDy, i64 0, i64 2), align 8
    ret i32 %0
; CHECK-LABEL: define i32 @AccessGlobDy_0_2
; CHECK: __asan_report
; CHECK: ret i32
}

; GlobEx is an external global -- can't optimize.
define i32 @AccessGlobEx_0_2() sanitize_address {
entry:
    %0 = load i32, ptr getelementptr inbounds ([10 x i32], ptr @GlobEx, i64 0, i64 2), align 8
    ret i32 %0
; CHECK-LABEL: define i32 @AccessGlobEx_0_2
; CHECK: __asan_report
; CHECK: ret i32
}

; CHECK-LABEL: define internal void @asan.module_ctor
; CHECK-NOT: ret
; CHECK: call void @__asan_register_elf_globals
; CHECK: ret

; CHECK-LABEL: define internal void @asan.module_dtor
; CHECK-NOT: ret
; CHECK: call void @__asan_unregister_elf_globals
; CHECK: ret
