; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s

define i8* @test_memcpy1(i8* %P, i8* %Q) {
  ; CHECK: test_memcpy
  call void @llvm.memcpy.element.atomic.p0i8.p0i8(i8* align 4 %P, i8* align 4 %Q, i64 1, i32 1)
  ret i8* %P
  ; CHECK-DAG: movl $1, %edx
  ; CHECK-DAG: movl $1, %ecx
  ; CHECK: __llvm_memcpy_element_atomic_1
}

define i8* @test_memcpy2(i8* %P, i8* %Q) {
  ; CHECK: test_memcpy2
  call void @llvm.memcpy.element.atomic.p0i8.p0i8(i8* align 4 %P, i8* align 4 %Q, i64 2, i32 2)
  ret i8* %P
  ; CHECK-DAG: movl $2, %edx
  ; CHECK-DAG: movl $2, %ecx
  ; CHECK: __llvm_memcpy_element_atomic_2
}

define i8* @test_memcpy4(i8* %P, i8* %Q) {
  ; CHECK: test_memcpy4
  call void @llvm.memcpy.element.atomic.p0i8.p0i8(i8* align 4 %P, i8* align 4 %Q, i64 4, i32 4)
  ret i8* %P
  ; CHECK-DAG: movl $4, %edx
  ; CHECK-DAG: movl $4, %ecx
  ; CHECK: __llvm_memcpy_element_atomic_4
}

define i8* @test_memcpy8(i8* %P, i8* %Q) {
  ; CHECK: test_memcpy8
  call void @llvm.memcpy.element.atomic.p0i8.p0i8(i8* align 8 %P, i8* align 8 %Q, i64 8, i32 8)
  ret i8* %P
  ; CHECK-DAG: movl $8, %edx
  ; CHECK-DAG: movl $8, %ecx
  ; CHECK: __llvm_memcpy_element_atomic_8
}

define i8* @test_memcpy16(i8* %P, i8* %Q) {
  ; CHECK: test_memcpy16
  call void @llvm.memcpy.element.atomic.p0i8.p0i8(i8* align 16 %P, i8* align 16 %Q, i64 16, i32 16)
  ret i8* %P
  ; CHECK-DAG: movl $16, %edx
  ; CHECK-DAG: movl $16, %ecx
  ; CHECK: __llvm_memcpy_element_atomic_16
}

define void @test_memcpy_args(i8** %Storage) {
  ; CHECK: test_memcpy_args
  %Dst = load i8*, i8** %Storage
  %Src.addr = getelementptr i8*, i8** %Storage, i64 1
  %Src = load i8*, i8** %Src.addr

  ; First argument
  ; CHECK-DAG: movq (%rdi), [[REG1:%r.+]]
  ; CHECK-DAG: movq [[REG1]], %rdi
  ; Second argument
  ; CHECK-DAG: movq 8(%rdi), %rsi
  ; Third argument
  ; CHECK-DAG: movl $4, %edx
  ; Fourth argument
  ; CHECK-DAG: movl $4, %ecx
  ; CHECK: __llvm_memcpy_element_atomic_4
  call void @llvm.memcpy.element.atomic.p0i8.p0i8(i8* align 4 %Dst, i8* align 4 %Src, i64 4, i32 4)
  ret void
}

declare void @llvm.memcpy.element.atomic.p0i8.p0i8(i8* nocapture, i8* nocapture, i64, i32) nounwind
