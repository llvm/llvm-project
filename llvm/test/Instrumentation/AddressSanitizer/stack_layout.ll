; Test the ASan's stack layout.
; More tests in tests/Transforms/Utils/ASanStackFrameLayoutTest.cpp
; RUN: opt < %s -asan -asan-module -asan-stack-dynamic-alloca=0 -asan-use-after-scope -S -enable-new-pm=0 \
; RUN:     | FileCheck %s --check-prefixes=CHECK,CHECK-STATIC
; RUN: opt < %s -passes='asan-pipeline' -asan-stack-dynamic-alloca=0 -asan-use-after-scope -S \
; RUN:     | FileCheck %s --check-prefixes=CHECK,CHECK-STATIC
; RUN: opt < %s -asan -asan-module -asan-stack-dynamic-alloca=1 -asan-use-after-scope -S -enable-new-pm=0 \
; RUN:     | FileCheck %s --check-prefixes=CHECK,CHECK-DYNAMIC
; RUN: opt < %s -passes='asan-pipeline' -asan-stack-dynamic-alloca=1 -asan-use-after-scope -S \
; RUN:     | FileCheck %s --check-prefixes=CHECK,CHECK-DYNAMIC

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @Use(i8*)
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) nounwind

; CHECK: private unnamed_addr constant{{.*}}3 32 10 3 XXX 64 20 3 YYY 128 30 3 ZZZ\0
; CHECK: private unnamed_addr constant{{.*}}3 32 5 3 AAA 64 55 3 BBB 160 555 3 CCC\0
; CHECK: private unnamed_addr constant{{.*}}3 256 128 3 CCC 448 128 3 BBB 608 128 3 AAA\0
; CHECK: private unnamed_addr constant{{.*}}2 32 4 3 AAA 48 4 5 BBB:7\0

define void @Func1() sanitize_address {
entry:
; CHECK-LABEL: Func1

; CHECK-STATIC: alloca [192 x i8]
; CHECK-STATIC: %asan_local_stack_base = alloca i64
; CHECK-DYNAMIC: alloca i8, i64 192

; CHECK-NOT: alloca
; CHECK: ret void
  %XXX = alloca [10 x i8], align 1
  %YYY = alloca [20 x i8], align 1
  %ZZZ = alloca [30 x i8], align 1
  %arr1.ptr = bitcast [10 x i8]* %XXX to i8*
  store volatile i8 0, i8* %arr1.ptr
  %arr2.ptr = bitcast [20 x i8]* %YYY to i8*
  store volatile i8 0, i8* %arr2.ptr
  %arr3.ptr = bitcast [30 x i8]* %ZZZ to i8*
  store volatile i8 0, i8* %arr3.ptr
  ret void
}

define void @Func2() sanitize_address {
entry:
; CHECK-LABEL: Func2

; CHECK-STATIC: alloca [864 x i8]
; CHECK-STATIC: %asan_local_stack_base = alloca i64
; CHECK-DYNAMIC: alloca i8, i64 864

; CHECK-NOT: alloca
; CHECK: ret void
  %AAA = alloca [5 x i8], align 1
  %BBB = alloca [55 x i8], align 1
  %CCC = alloca [555 x i8], align 1
  %arr1.ptr = bitcast [5 x i8]* %AAA to i8*
  store volatile i8 0, i8* %arr1.ptr
  %arr2.ptr = bitcast [55 x i8]* %BBB to i8*
  store volatile i8 0, i8* %arr2.ptr
  %arr3.ptr = bitcast [555 x i8]* %CCC to i8*
  store volatile i8 0, i8* %arr3.ptr
  ret void
}

; Check that we reorder vars according to alignment and handle large alignments.
define void @Func3() sanitize_address {
entry:
; CHECK-LABEL: Func3

; CHECK-STATIC: alloca [768 x i8]
; CHECK-STATIC: %asan_local_stack_base = alloca i64
; CHECK-DYNAMIC: alloca i8, i64 768

; CHECK-NOT: alloca
; CHECK: ret void
  %AAA = alloca [128 x i8], align 16
  %BBB = alloca [128 x i8], align 64
  %CCC = alloca [128 x i8], align 256
  %arr1.ptr = bitcast [128 x i8]* %AAA to i8*
  store volatile i8 0, i8* %arr1.ptr
  %arr2.ptr = bitcast [128 x i8]* %BBB to i8*
  store volatile i8 0, i8* %arr2.ptr
  %arr3.ptr = bitcast [128 x i8]* %CCC to i8*
  store volatile i8 0, i8* %arr3.ptr
  ret void
}

; Check that line numbers are attached to variable names if variable
; in the same file as a function.
define void @Func5() sanitize_address #0 !dbg !11 {
  %AAA = alloca i32, align 4  ; File is not the same as !11
  %BBB = alloca i32, align 4  ; File is the same as !11
  %BBB.ptr = bitcast i32* %BBB to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %BBB.ptr), !dbg !12
  store volatile i32 5, i32* %BBB, align 4
  %AAA.ptr = bitcast i32* %AAA to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %AAA.ptr), !dbg !14
  store volatile i32 3, i32* %AAA, align 4
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %AAA.ptr), !dbg !17
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %BBB.ptr), !dbg !18
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1)
!1 = !DIFile(filename: "../file1.c", directory: "/")
!11 = distinct !DISubprogram(name: "Func5", scope: !1, file: !1, line: 6, unit: !0)
!12 = !DILocation(line: 7, column: 3, scope: !11)
!18 = !DILocation(line: 10, column: 1, scope: !11)

!21 = !DIFile(filename: "../file2.c", directory: "/")
!6 = distinct !DISubprogram(name: "Func4", scope: !1, file: !21, line: 2, unit: !0)
!15 = distinct !DILocation(line: 8, column: 3, scope: !11)
!14 = !DILocation(line: 3, column: 3, scope: !6, inlinedAt: !15)
!17 = !DILocation(line: 4, column: 1, scope: !6, inlinedAt: !15)
