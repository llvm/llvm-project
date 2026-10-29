; RUN: split-file %s %t

;--- test0.ll
; RUN: not opt -S -passes=verify -disable-output 2>&1 < %t/test0.ll | FileCheck %t/test0.ll

; CHECK: intrinsic return type expected vector with 16 elements, but got i32
; CHECK-NEXT: declare i32 @llvm.aarch64.neon.pmull64(i64, i64)

; Correct return type is <16 x i8>
declare i32 @llvm.aarch64.neon.pmull64(i64, i64)

; CHECK: intrinsic return type expected literal non-packed struct with 2 elements, but got void
; CHECK-NEXT: declare void @llvm.nvvm.elect.sync(i32)

; Expected return type is { i32, i1 }.
declare void @llvm.nvvm.elect.sync(i32)

;--- test1.ll
; RUN: not opt -S -passes=verify -disable-output 2>&1 < %t/test1.ll | FileCheck %t/test1.ll

; CHECK: intrinsic return vector element type expected i8, but got i32
; CHECK-NEXT: declare <16 x i32> @llvm.aarch64.neon.pmull64(i64, i64)
declare <16 x i32> @llvm.aarch64.neon.pmull64(i64, i64)

; CHECK: intrinsic return struct element 1 type expected i1, but got i2
; CHECK-NEXT: declare { i32, i2 } @llvm.nvvm.elect.sync(i32)

; Expected return type is { i32, i1 }.
declare { i32, i2 } @llvm.nvvm.elect.sync(i32)

;--- test2.ll
; RUN: not opt -S -passes=verify -disable-output 2>&1 < %t/test2.ll | FileCheck %t/test2.ll

; CHECK: intrinsic return type expected vector with 16 elements, but got <vscale x 16 x i32>
; CHECK-NEXT: declare <vscale x 16 x i32> @llvm.aarch64.neon.pmull64(i64, i64)
declare <vscale x 16 x i32> @llvm.aarch64.neon.pmull64(i64, i64)

; CHECK: intrinsic return type expected literal non-packed struct with 2 elements, but got { i32, i1, i1 }
; CHECK-NEXT: declare { i32, i1, i1 } @llvm.nvvm.elect.sync(i32)

; Expected return type is { i32, i1 }.
declare { i32, i1, i1 } @llvm.nvvm.elect.sync(i32)
