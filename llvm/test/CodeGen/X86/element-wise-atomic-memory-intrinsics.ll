; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s

define i8* @test_memcpy1(i8* %P, i8* %Q) {
  ; CHECK: test_memcpy
  call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i32(i8* align 4 %P, i8* align 4 %Q, i32 1024, i32 1)
  ret i8* %P
  ; 3rd arg (%edx) -- length
  ; CHECK-DAG: movl $1024, %edx
  ; CHECK: __llvm_memcpy_element_unordered_atomic_1
}

define i8* @test_memcpy2(i8* %P, i8* %Q) {
  ; CHECK: test_memcpy2
  call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i32(i8* align 4 %P, i8* align 4 %Q, i32 1024, i32 2)
  ret i8* %P
  ; 3rd arg (%edx) -- length
  ; CHECK-DAG: movl $1024, %edx
  ; CHECK: __llvm_memcpy_element_unordered_atomic_2
}

define i8* @test_memcpy4(i8* %P, i8* %Q) {
  ; CHECK: test_memcpy4
  call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i32(i8* align 4 %P, i8* align 4 %Q, i32 1024, i32 4)
  ret i8* %P
  ; 3rd arg (%edx) -- length
  ; CHECK-DAG: movl $1024, %edx
  ; CHECK: __llvm_memcpy_element_unordered_atomic_4
}

define i8* @test_memcpy8(i8* %P, i8* %Q) {
  ; CHECK: test_memcpy8
  call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i32(i8* align 8 %P, i8* align 8 %Q, i32 1024, i32 8)
  ret i8* %P
  ; 3rd arg (%edx) -- length
  ; CHECK-DAG: movl $1024, %edx
  ; CHECK: __llvm_memcpy_element_unordered_atomic_8
}

define i8* @test_memcpy16(i8* %P, i8* %Q) {
  ; CHECK: test_memcpy16
  call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i32(i8* align 16 %P, i8* align 16 %Q, i32 1024, i32 16)
  ret i8* %P
  ; 3rd arg (%edx) -- length
  ; CHECK-DAG: movl $1024, %edx
  ; CHECK: __llvm_memcpy_element_unordered_atomic_16
}

define void @test_memcpy_args(i8** %Storage) {
  ; CHECK: test_memcpy_args
  %Dst = load i8*, i8** %Storage
  %Src.addr = getelementptr i8*, i8** %Storage, i64 1
  %Src = load i8*, i8** %Src.addr

  ; 1st arg (%rdi)
  ; CHECK-DAG: movq (%rdi), [[REG1:%r.+]]
  ; CHECK-DAG: movq [[REG1]], %rdi
  ; 2nd arg (%rsi)
  ; CHECK-DAG: movq 8(%rdi), %rsi
  ; 3rd arg (%edx) -- length
  ; CHECK-DAG: movl $1024, %edx
  ; CHECK: __llvm_memcpy_element_unordered_atomic_4
  call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i32(i8* align 4 %Dst, i8* align 4 %Src, i32 1024, i32 4)
  ret void
}

define i8* @test_memmove1(i8* %P, i8* %Q) {
  ; CHECK: test_memmove
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 4 %P, i8* align 4 %Q, i32 1024, i32 1)
  ret i8* %P
  ; 3rd arg (%edx) -- length
  ; CHECK-DAG: movl $1024, %edx
  ; CHECK: __llvm_memmove_element_unordered_atomic_1
}

define i8* @test_memmove2(i8* %P, i8* %Q) {
  ; CHECK: test_memmove2
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 4 %P, i8* align 4 %Q, i32 1024, i32 2)
  ret i8* %P
  ; 3rd arg (%edx) -- length
  ; CHECK-DAG: movl $1024, %edx
  ; CHECK: __llvm_memmove_element_unordered_atomic_2
}

define i8* @test_memmove4(i8* %P, i8* %Q) {
  ; CHECK: test_memmove4
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 4 %P, i8* align 4 %Q, i32 1024, i32 4)
  ret i8* %P
  ; 3rd arg (%edx) -- length
  ; CHECK-DAG: movl $1024, %edx
  ; CHECK: __llvm_memmove_element_unordered_atomic_4
}

define i8* @test_memmove8(i8* %P, i8* %Q) {
  ; CHECK: test_memmove8
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 8 %P, i8* align 8 %Q, i32 1024, i32 8)
  ret i8* %P
  ; 3rd arg (%edx) -- length
  ; CHECK-DAG: movl $1024, %edx
  ; CHECK: __llvm_memmove_element_unordered_atomic_8
}

define i8* @test_memmove16(i8* %P, i8* %Q) {
  ; CHECK: test_memmove16
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 16 %P, i8* align 16 %Q, i32 1024, i32 16)
  ret i8* %P
  ; 3rd arg (%edx) -- length
  ; CHECK-DAG: movl $1024, %edx
  ; CHECK: __llvm_memmove_element_unordered_atomic_16
}

define void @test_memmove_args(i8** %Storage) {
  ; CHECK: test_memmove_args
  %Dst = load i8*, i8** %Storage
  %Src.addr = getelementptr i8*, i8** %Storage, i64 1
  %Src = load i8*, i8** %Src.addr

  ; 1st arg (%rdi)
  ; CHECK-DAG: movq (%rdi), [[REG1:%r.+]]
  ; CHECK-DAG: movq [[REG1]], %rdi
  ; 2nd arg (%rsi)
  ; CHECK-DAG: movq 8(%rdi), %rsi
  ; 3rd arg (%edx) -- length
  ; CHECK-DAG: movl $1024, %edx
  ; CHECK: __llvm_memmove_element_unordered_atomic_4
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 4 %Dst, i8* align 4 %Src, i32 1024, i32 4)
  ret void
}

define i8* @test_memset1(i8* %P, i8 %V) {
  ; CHECK: test_memset
  call void @llvm.memset.element.unordered.atomic.p0i8.i32(i8* align 1 %P, i8 %V, i32 1024, i32 1)
  ret i8* %P
  ; 3rd arg (%edx) -- length
  ; CHECK-DAG: movl $1024, %edx
  ; CHECK: __llvm_memset_element_unordered_atomic_1
}

define i8* @test_memset2(i8* %P, i8 %V) {
  ; CHECK: test_memset2
  call void @llvm.memset.element.unordered.atomic.p0i8.i32(i8* align 2 %P, i8 %V, i32 1024, i32 2)
  ret i8* %P
  ; 3rd arg (%edx) -- length
  ; CHECK-DAG: movl $1024, %edx
  ; CHECK: __llvm_memset_element_unordered_atomic_2
}

define i8* @test_memset4(i8* %P, i8 %V) {
  ; CHECK: test_memset4
  call void @llvm.memset.element.unordered.atomic.p0i8.i32(i8* align 4 %P, i8 %V, i32 1024, i32 4)
  ret i8* %P
  ; 3rd arg (%edx) -- length
  ; CHECK-DAG: movl $1024, %edx
  ; CHECK: __llvm_memset_element_unordered_atomic_4
}

define i8* @test_memset8(i8* %P, i8 %V) {
  ; CHECK: test_memset8
  call void @llvm.memset.element.unordered.atomic.p0i8.i32(i8* align 8 %P, i8 %V, i32 1024, i32 8)
  ret i8* %P
  ; 3rd arg (%edx) -- length
  ; CHECK-DAG: movl $1024, %edx
  ; CHECK: __llvm_memset_element_unordered_atomic_8
}

define i8* @test_memset16(i8* %P, i8 %V) {
  ; CHECK: test_memset16
  call void @llvm.memset.element.unordered.atomic.p0i8.i32(i8* align 16 %P, i8 %V, i32 1024, i32 16)
  ret i8* %P
  ; 3rd arg (%edx) -- length
  ; CHECK-DAG: movl $1024, %edx
  ; CHECK: __llvm_memset_element_unordered_atomic_16
}

define void @test_memset_args(i8** %Storage, i8* %V) {
  ; CHECK: test_memset_args
  %Dst = load i8*, i8** %Storage
  %Val = load i8, i8* %V

  ; 1st arg (%rdi)
  ; CHECK-DAG: movq (%rdi), %rdi
  ; 2nd arg (%rsi)
  ; CHECK-DAG: movzbl (%rsi), %esi
  ; 3rd arg (%edx) -- length
  ; CHECK-DAG: movl $1024, %edx
  ; CHECK: __llvm_memset_element_unordered_atomic_4
  call void @llvm.memset.element.unordered.atomic.p0i8.i32(i8* align 4 %Dst, i8 %Val, i32 1024, i32 4)
  ret void
}

declare void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32) nounwind
declare void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32) nounwind
declare void @llvm.memset.element.unordered.atomic.p0i8.i32(i8* nocapture, i8, i32, i32) nounwind
