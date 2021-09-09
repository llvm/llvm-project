; RUN: opt < %s -passes=asan-pipeline -asan-instrumentation-with-call-threshold=0 \
; RUN:   -asan-optimize-callbacks -S | FileCheck %s --check-prefixes=LOAD,STORE
; RUN: opt < %s -passes=asan-pipeline -asan-instrumentation-with-call-threshold=0 \
; RUN:   -asan-optimize-callbacks --asan-kernel -S | \
; RUN:   FileCheck %s --check-prefixes=LOAD-KERNEL,STORE-KERNEL

target triple = "x86_64-unknown-linux-gnu"

define void @load(i8* %p1, i16* %p2, i32* %p4, i64* %p8, i128* %p16)
sanitize_address {
  %n1 = load i8, i8* %p1, align 1
  %n2 = load i16, i16* %p2, align 2
  %n4 = load i32, i32* %p4, align 4
  %n8 = load i64, i64* %p8, align 8
  %n16 = load i128, i128* %p16, align 16
; LOAD:      call void @llvm.asan.check.memaccess(i8* %p1, i32 0)
; LOAD-NEXT: %n1 = load i8, i8* %p1, align 1
; LOAD-NEXT: %1 = bitcast i16* %p2 to i8*
; LOAD-NEXT: call void @llvm.asan.check.memaccess(i8* %1, i32 2)
; LOAD-NEXT: %n2 = load i16, i16* %p2, align 2
; LOAD-NEXT: %2 = bitcast i32* %p4 to i8*
; LOAD-NEXT: call void @llvm.asan.check.memaccess(i8* %2, i32 4)
; LOAD-NEXT: %n4 = load i32, i32* %p4, align 4
; LOAD-NEXT: %3 = bitcast i64* %p8 to i8*
; LOAD-NEXT: call void @llvm.asan.check.memaccess(i8* %3, i32 6)
; LOAD-NEXT: %n8 = load i64, i64* %p8, align 8
; LOAD-NEXT: %4 = bitcast i128* %p16 to i8*
; LOAD-NEXT: call void @llvm.asan.check.memaccess(i8* %4, i32 8)
; LOAD-NEXT: %n16 = load i128, i128* %p16, align 16

; LOAD-KERNEL:      call void @llvm.asan.check.memaccess(i8* %p1, i32 1)
; LOAD-KERNEL-NEXT: %n1 = load i8, i8* %p1, align 1
; LOAD-KERNEL-NEXT: %1 = bitcast i16* %p2 to i8*
; LOAD-KERNEL-NEXT: call void @llvm.asan.check.memaccess(i8* %1, i32 3)
; LOAD-KERNEL-NEXT: %n2 = load i16, i16* %p2, align 2
; LOAD-KERNEL-NEXT: %2 = bitcast i32* %p4 to i8*
; LOAD-KERNEL-NEXT: call void @llvm.asan.check.memaccess(i8* %2, i32 5)
; LOAD-KERNEL-NEXT: %n4 = load i32, i32* %p4, align 4
; LOAD-KERNEL-NEXT: %3 = bitcast i64* %p8 to i8*
; LOAD-KERNEL-NEXT: call void @llvm.asan.check.memaccess(i8* %3, i32 7)
; LOAD-KERNEL-NEXT: %n8 = load i64, i64* %p8, align 8
; LOAD-KERNEL-NEXT: %4 = bitcast i128* %p16 to i8*
; LOAD-KERNEL-NEXT: call void @llvm.asan.check.memaccess(i8* %4, i32 9)
; LOAD-KERNEL-NEXT: %n16 = load i128, i128* %p16, align 16
  ret void
}

define void @store(i8* %p1, i16* %p2, i32* %p4, i64* %p8, i128* %p16)
sanitize_address {
  store i8 0, i8* %p1, align 1
  store i16 0, i16* %p2, align 2
  store i32 0, i32* %p4, align 4
  store i64 0, i64* %p8, align 8
  store i128 0, i128* %p16, align 16
; STORE:      call void @llvm.asan.check.memaccess(i8* %p1, i32 32)
; STORE-NEXT: store i8 0, i8* %p1, align 1
; STORE-NEXT: %1 = bitcast i16* %p2 to i8*
; STORE-NEXT: call void @llvm.asan.check.memaccess(i8* %1, i32 34)
; STORE-NEXT: store i16 0, i16* %p2, align 2
; STORE-NEXT: %2 = bitcast i32* %p4 to i8*
; STORE-NEXT: call void @llvm.asan.check.memaccess(i8* %2, i32 36)
; STORE-NEXT: store i32 0, i32* %p4, align 4
; STORE-NEXT: %3 = bitcast i64* %p8 to i8*
; STORE-NEXT: call void @llvm.asan.check.memaccess(i8* %3, i32 38)
; STORE-NEXT: store i64 0, i64* %p8, align 8
; STORE-NEXT: %4 = bitcast i128* %p16 to i8*
; STORE-NEXT: call void @llvm.asan.check.memaccess(i8* %4, i32 40)
; STORE-NEXT: store i128 0, i128* %p16, align 16

; STORE-KERNEL:      call void @llvm.asan.check.memaccess(i8* %p1, i32 33)
; STORE-KERNEL-NEXT: store i8 0, i8* %p1, align 1
; STORE-KERNEL-NEXT: %1 = bitcast i16* %p2 to i8*
; STORE-KERNEL-NEXT: call void @llvm.asan.check.memaccess(i8* %1, i32 35)
; STORE-KERNEL-NEXT: store i16 0, i16* %p2, align 2
; STORE-KERNEL-NEXT: %2 = bitcast i32* %p4 to i8*
; STORE-KERNEL-NEXT: call void @llvm.asan.check.memaccess(i8* %2, i32 37)
; STORE-KERNEL-NEXT: store i32 0, i32* %p4, align 4
; STORE-KERNEL-NEXT: %3 = bitcast i64* %p8 to i8*
; STORE-KERNEL-NEXT: call void @llvm.asan.check.memaccess(i8* %3, i32 39)
; STORE-KERNEL-NEXT: store i64 0, i64* %p8, align 8
; STORE-KERNEL-NEXT: %4 = bitcast i128* %p16 to i8*
; STORE-KERNEL-NEXT: call void @llvm.asan.check.memaccess(i8* %4, i32 41)
; STORE-KERNEL-NEXT: store i128 0, i128* %p16, align 16
; STORE-KERNEL-NEXT: ret void
  ret void
}
