; RUN: opt < %s -passes=asan -asan-instrumentation-with-call-threshold=0 -asan-optimize-callbacks -S | FileCheck %s --check-prefixes=LOAD,STORE
; RUN: opt < %s -passes=asan -asan-instrumentation-with-call-threshold=0 -asan-optimize-callbacks --asan-kernel -S | FileCheck %s --check-prefixes=LOAD-KERNEL,STORE-KERNEL

target triple = "x86_64-unknown-linux-gnu"

define void @load(ptr %p1, ptr %p2, ptr %p4, ptr %p8, ptr %p16)
sanitize_address {
  %n1 = load i8, ptr %p1, align 1
  %n2 = load i16, ptr %p2, align 2
  %n4 = load i32, ptr %p4, align 4
  %n8 = load i64, ptr %p8, align 8
  %n16 = load i128, ptr %p16, align 16
; LOAD:      call void @llvm.asan.check.memaccess(ptr %p1, i32 0)
; LOAD-NEXT: %n1 = load i8, ptr %p1, align 1
; LOAD-NEXT: call void @llvm.asan.check.memaccess(ptr %p2, i32 2)
; LOAD-NEXT: %n2 = load i16, ptr %p2, align 2
; LOAD-NEXT: call void @llvm.asan.check.memaccess(ptr %p4, i32 4)
; LOAD-NEXT: %n4 = load i32, ptr %p4, align 4
; LOAD-NEXT: call void @llvm.asan.check.memaccess(ptr %p8, i32 6)
; LOAD-NEXT: %n8 = load i64, ptr %p8, align 8
; LOAD-NEXT: call void @llvm.asan.check.memaccess(ptr %p16, i32 8)
; LOAD-NEXT: %n16 = load i128, ptr %p16, align 16

; LOAD-KERNEL:      call void @llvm.asan.check.memaccess(ptr %p1, i32 1)
; LOAD-KERNEL-NEXT: %n1 = load i8, ptr %p1, align 1
; LOAD-KERNEL-NEXT: call void @llvm.asan.check.memaccess(ptr %p2, i32 3)
; LOAD-KERNEL-NEXT: %n2 = load i16, ptr %p2, align 2
; LOAD-KERNEL-NEXT: call void @llvm.asan.check.memaccess(ptr %p4, i32 5)
; LOAD-KERNEL-NEXT: %n4 = load i32, ptr %p4, align 4
; LOAD-KERNEL-NEXT: call void @llvm.asan.check.memaccess(ptr %p8, i32 7)
; LOAD-KERNEL-NEXT: %n8 = load i64, ptr %p8, align 8
; LOAD-KERNEL-NEXT: call void @llvm.asan.check.memaccess(ptr %p16, i32 9)
; LOAD-KERNEL-NEXT: %n16 = load i128, ptr %p16, align 16
  ret void
}

define void @store(ptr %p1, ptr %p2, ptr %p4, ptr %p8, ptr %p16)
sanitize_address {
  store i8 0, ptr %p1, align 1
  store i16 0, ptr %p2, align 2
  store i32 0, ptr %p4, align 4
  store i64 0, ptr %p8, align 8
  store i128 0, ptr %p16, align 16
; STORE:      call void @llvm.asan.check.memaccess(ptr %p1, i32 32)
; STORE-NEXT: store i8 0, ptr %p1, align 1
; STORE-NEXT: call void @llvm.asan.check.memaccess(ptr %p2, i32 34)
; STORE-NEXT: store i16 0, ptr %p2, align 2
; STORE-NEXT: call void @llvm.asan.check.memaccess(ptr %p4, i32 36)
; STORE-NEXT: store i32 0, ptr %p4, align 4
; STORE-NEXT: call void @llvm.asan.check.memaccess(ptr %p8, i32 38)
; STORE-NEXT: store i64 0, ptr %p8, align 8
; STORE-NEXT: call void @llvm.asan.check.memaccess(ptr %p16, i32 40)
; STORE-NEXT: store i128 0, ptr %p16, align 16

; STORE-KERNEL:      call void @llvm.asan.check.memaccess(ptr %p1, i32 33)
; STORE-KERNEL-NEXT: store i8 0, ptr %p1, align 1
; STORE-KERNEL-NEXT: call void @llvm.asan.check.memaccess(ptr %p2, i32 35)
; STORE-KERNEL-NEXT: store i16 0, ptr %p2, align 2
; STORE-KERNEL-NEXT: call void @llvm.asan.check.memaccess(ptr %p4, i32 37)
; STORE-KERNEL-NEXT: store i32 0, ptr %p4, align 4
; STORE-KERNEL-NEXT: call void @llvm.asan.check.memaccess(ptr %p8, i32 39)
; STORE-KERNEL-NEXT: store i64 0, ptr %p8, align 8
; STORE-KERNEL-NEXT: call void @llvm.asan.check.memaccess(ptr %p16, i32 41)
; STORE-KERNEL-NEXT: store i128 0, ptr %p16, align 16
; STORE-KERNEL-NEXT: ret void
  ret void
}
