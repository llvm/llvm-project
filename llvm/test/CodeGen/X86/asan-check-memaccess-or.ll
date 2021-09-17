; RUN: llc < %s | FileCheck %s

target triple = "x86_64-pc-win"

define void @load1(i8* nocapture readonly %x) {
; CHECK:              pushq   %rax
; CHECK-NOT:          push    %rbp
; CHECK:              callq   __asan_check_load1_rn[[RN1:.*]]
; CHECK:              callq   __asan_check_store1_rn[[RN1]]
; CHECK-NOT:          pop     %rbp
; CHECK:              popq    %rax
  call void @llvm.asan.check.memaccess(i8* %x, i32 0)
  call void @llvm.asan.check.memaccess(i8* %x, i32 32)
  ret void
}

define void @load2(i16* nocapture readonly %x) {
; CHECK:              pushq   %rax
; CHECK-NOT:          push    %rbp
; CHECK:              callq   __asan_check_load2_rn[[RN2:.*]]
; CHECK:              callq   __asan_check_store2_rn[[RN2]]
; CHECK-NOT:          pop     %rbp
; CHECK:              popq    %rax
  %1 = ptrtoint i16* %x to i64
  %2 = bitcast i16* %x to i8*
  call void @llvm.asan.check.memaccess(i8* %2, i32 2)
  call void @llvm.asan.check.memaccess(i8* %2, i32 34)
  ret void
}

define void @load4(i32* nocapture readonly %x) {
; CHECK:              pushq   %rax
; CHECK-NOT:          push    %rbp
; CHECK:              callq   __asan_check_load4_rn[[RN4:.*]]
; CHECK:              callq   __asan_check_store4_rn[[RN4]]
; CHECK-NOT:          pop     %rbp
; CHECK:              popq    %rax
  %1 = ptrtoint i32* %x to i64
  %2 = bitcast i32* %x to i8*
  call void @llvm.asan.check.memaccess(i8* %2, i32 4)
  call void @llvm.asan.check.memaccess(i8* %2, i32 36)
  ret void
}
define void @load8(i64* nocapture readonly %x) {
; CHECK:              pushq   %rax
; CHECK-NOT:          push    %rbp
; CHECK:              callq   __asan_check_load8_rn[[RN8:.*]]
; CHECK:              callq   __asan_check_store8_rn[[RN8]]
; CHECK-NOT:          pop     %rbp
; CHECK:              popq    %rax
  %1 = ptrtoint i64* %x to i64
  %2 = bitcast i64* %x to i8*
  call void @llvm.asan.check.memaccess(i8* %2, i32 6)
  call void @llvm.asan.check.memaccess(i8* %2, i32 38)
  ret void
}

define void @load16(i128* nocapture readonly %x) {
; CHECK:              pushq   %rax
; CHECK-NOT:          push    %rbp
; CHECK:              callq   __asan_check_load16_rn[[RN16:.*]]
; CHECK:              callq   __asan_check_store16_rn[[RN16]]
; CHECK-NOT:          pop     %rbp
; CHECK:              popq    %rax
  %1 = ptrtoint i128* %x to i64
  %2 = bitcast i128* %x to i8*
  call void @llvm.asan.check.memaccess(i8* %2, i32 8)
  call void @llvm.asan.check.memaccess(i8* %2, i32 40)
  ret void
}

; CHECK:              .type   __asan_check_load1_rn[[RN1]],@function
; CHECK-NEXT:         .weak   __asan_check_load1_rn[[RN1]]
; CHECK-NEXT:         .hidden __asan_check_load1_rn[[RN1]]
; CHECK-NEXT: __asan_check_load1_rn[[RN1]]:
; CHECK-NEXT:         movq    [[REG:.*]], %r8
; CHECK-NEXT:         shrq    $3, %r8
; CHECK-NEXT:         orq     $17592186044416, %r8{{.*}}
; CHECK-NEXT:         movb    (%r8), %r8b
; CHECK-NEXT:         testb   %r8b, %r8b
; CHECK-NEXT:         jne     [[EXTRA:.*]]
; CHECK-NEXT: [[RET:.*]]:
; CHECK-NEXT:         retq
; CHECK-NEXT: [[EXTRA]]:
; CHECK-NEXT:         pushq   %rcx
; CHECK-NEXT:         movq    [[REG]], %rcx
; CHECK-NEXT:         andl    $7, %ecx
; CHECK-NEXT:         cmpl    %r8d, %ecx
; CHECK-NEXT:         popq    %rcx
; CHECK-NEXT:         jl      [[RET]]
; CHECK-NEXT:         movq    [[REG:.*]], %rdi
; CHECK-NEXT:         jmp     __asan_report_load1

; CHECK:              .type   __asan_check_load2_rn[[RN2]],@function
; CHECK-NEXT:         .weak   __asan_check_load2_rn[[RN2]]
; CHECK-NEXT:         .hidden __asan_check_load2_rn[[RN2]]
; CHECK-NEXT: __asan_check_load2_rn[[RN2]]:
; CHECK-NEXT:         movq    [[REG:.*]], %r8
; CHECK-NEXT:         shrq    $3, %r8
; CHECK-NEXT:         orq     $17592186044416, %r8{{.*}}
; CHECK-NEXT:         movb    (%r8), %r8b
; CHECK-NEXT:         testb   %r8b, %r8b
; CHECK-NEXT:         jne     [[EXTRA:.*]]
; CHECK-NEXT: [[RET:.*]]:
; CHECK-NEXT:         retq
; CHECK-NEXT: [[EXTRA]]:
; CHECK-NEXT:         pushq   %rcx
; CHECK-NEXT:         movq    [[REG]], %rcx
; CHECK-NEXT:         andl    $7, %ecx
; CHECK-NEXT:         addl    $1, %ecx
; CHECK-NEXT:         cmpl    %r8d, %ecx
; CHECK-NEXT:         popq    %rcx
; CHECK-NEXT:         jl      [[RET]]
; CHECK-NEXT:         movq    [[REG:.*]], %rdi
; CHECK-NEXT:         jmp     __asan_report_load2

; CHECK:              .type   __asan_check_load4_rn[[RN4]],@function
; CHECK-NEXT:         .weak   __asan_check_load4_rn[[RN4]]
; CHECK-NEXT:         .hidden __asan_check_load4_rn[[RN4]]
; CHECK-NEXT: __asan_check_load4_rn[[RN4]]:
; CHECK-NEXT:         movq    [[REG:.*]], %r8
; CHECK-NEXT:         shrq    $3, %r8
; CHECK-NEXT:         orq     $17592186044416, %r8{{.*}}
; CHECK-NEXT:         movb    (%r8), %r8b
; CHECK-NEXT:         testb   %r8b, %r8b
; CHECK-NEXT:         jne     [[EXTRA:.*]]
; CHECK-NEXT: [[RET:.*]]:
; CHECK-NEXT:         retq
; CHECK-NEXT: [[EXTRA]]:
; CHECK-NEXT:         pushq   %rcx
; CHECK-NEXT:         movq    [[REG]], %rcx
; CHECK-NEXT:         andl    $7, %ecx
; CHECK-NEXT:         addl    $3, %ecx
; CHECK-NEXT:         cmpl    %r8d, %ecx
; CHECK-NEXT:         popq    %rcx
; CHECK-NEXT:         jl      [[RET]]
; CHECK-NEXT:         movq    [[REG:.*]], %rdi
; CHECK-NEXT:         jmp     __asan_report_load4

; CHECK:              .type   __asan_check_load8_rn[[RN8]],@function
; CHECK-NEXT:         .weak   __asan_check_load8_rn[[RN8]]
; CHECK-NEXT:         .hidden __asan_check_load8_rn[[RN8]]
; CHECK-NEXT: __asan_check_load8_rn[[RN8]]:
; CHECK-NEXT:         movq    [[REG:.*]], %r8
; CHECK-NEXT:         shrq    $3, %r8
; CHECK-NEXT:         orq     $17592186044416, %r8{{.*}}
; CHECK-NEXT:         cmpb    $0, (%r8)
; CHECK-NEXT:         jne     [[FAIL:.*]]
; CHECK-NEXT: [[RET:.*]]:
; CHECK-NEXT:         retq
; CHECK-NEXT: [[FAIL]]:
; CHECK-NEXT:         movq    [[REG:.*]], %rdi
; CHECK-NEXT:         jmp     __asan_report_load8

; CHECK:              .type   __asan_check_load16_rn[[RN16]],@function
; CHECK-NEXT:         .weak   __asan_check_load16_rn[[RN16]]
; CHECK-NEXT:         .hidden __asan_check_load16_rn[[RN16]]
; CHECK-NEXT: __asan_check_load16_rn[[RN16]]:
; CHECK-NEXT:         movq    [[REG:.*]], %r8
; CHECK-NEXT:         shrq    $3, %r8
; CHECK-NEXT:         orq     $17592186044416, %r8{{.*}}
; CHECK-NEXT:         cmpw    $0, (%r8)
; CHECK-NEXT:         jne     [[FAIL:.*]]
; CHECK-NEXT: [[RET:.*]]:
; CHECK-NEXT:         retq
; CHECK-NEXT: [[FAIL]]:
; CHECK-NEXT:         movq    [[REG:.*]], %rdi
; CHECK-NEXT:         jmp     __asan_report_load16

; CHECK:              .type   __asan_check_store1_rn[[RN1]],@function
; CHECK-NEXT:         .weak   __asan_check_store1_rn[[RN1]]
; CHECK-NEXT:         .hidden __asan_check_store1_rn[[RN1]]
; CHECK-NEXT: __asan_check_store1_rn[[RN1]]:
; CHECK-NEXT:         movq    [[REG:.*]], %r8
; CHECK-NEXT:         shrq    $3, %r8
; CHECK-NEXT:         orq     $17592186044416, %r8
; CHECK-NEXT:         movb    (%r8), %r8b
; CHECK-NEXT:         testb   %r8b, %r8b
; CHECK-NEXT:         jne     [[EXTRA:.*]]
; CHECK-NEXT: [[RET:.*]]:
; CHECK-NEXT:         retq
; CHECK-NEXT: [[EXTRA]]:
; CHECK-NEXT:         pushq   %rcx
; CHECK-NEXT:         movq    [[REG]], %rcx
; CHECK-NEXT:         andl    $7, %ecx
; CHECK-NEXT:         cmpl    %r8d, %ecx
; CHECK-NEXT:         popq    %rcx
; CHECK-NEXT:         jl      [[RET]]
; CHECK-NEXT:         movq    [[REG:.*]], %rdi
; CHECK-NEXT:         jmp     __asan_report_store1@PLT

; CHECK:              .type   __asan_check_store2_rn[[RN2]],@function
; CHECK-NEXT:         .weak   __asan_check_store2_rn[[RN2]]
; CHECK-NEXT:         .hidden __asan_check_store2_rn[[RN2]]
; CHECK-NEXT: __asan_check_store2_rn[[RN2]]:
; CHECK-NEXT:         movq    [[REG:.*]], %r8
; CHECK-NEXT:         shrq    $3, %r8
; CHECK-NEXT:         orq     $17592186044416, %r8
; CHECK-NEXT:         movb    (%r8), %r8b
; CHECK-NEXT:         testb   %r8b, %r8b
; CHECK-NEXT:         jne     [[EXTRA:.*]]
; CHECK-NEXT: [[RET:.*]]:
; CHECK-NEXT:         retq
; CHECK-NEXT: [[EXTRA]]:
; CHECK-NEXT:         pushq   %rcx
; CHECK-NEXT:         movq    [[REG]], %rcx
; CHECK-NEXT:         andl    $7, %ecx
; CHECK-NEXT:         addl    $1, %ecx
; CHECK-NEXT:         cmpl    %r8d, %ecx
; CHECK-NEXT:         popq    %rcx
; CHECK-NEXT:         jl      [[RET]]
; CHECK-NEXT:         movq    [[REG:.*]], %rdi
; CHECK-NEXT:         jmp     __asan_report_store2@PLT

; CHECK:              .type   __asan_check_store4_rn[[RN4]],@function
; CHECK-NEXT:         .weak   __asan_check_store4_rn[[RN4]]
; CHECK-NEXT:         .hidden __asan_check_store4_rn[[RN4]]
; CHECK-NEXT: __asan_check_store4_rn[[RN4]]:
; CHECK-NEXT:         movq    [[REG:.*]], %r8
; CHECK-NEXT:         shrq    $3, %r8
; CHECK-NEXT:         orq     $17592186044416, %r8
; CHECK-NEXT:         movb    (%r8), %r8b
; CHECK-NEXT:         testb   %r8b, %r8b
; CHECK-NEXT:         jne     [[EXTRA:.*]]
; CHECK-NEXT: [[RET:.*]]:
; CHECK-NEXT:         retq
; CHECK-NEXT: [[EXTRA]]:
; CHECK-NEXT:         pushq   %rcx
; CHECK-NEXT:         movq    [[REG]], %rcx
; CHECK-NEXT:         andl    $7, %ecx
; CHECK-NEXT:         addl    $3, %ecx
; CHECK-NEXT:         cmpl    %r8d, %ecx
; CHECK-NEXT:         popq    %rcx
; CHECK-NEXT:         jl      [[RET]]
; CHECK-NEXT:         movq    [[REG:.*]], %rdi
; CHECK-NEXT:         jmp     __asan_report_store4@PLT

; CHECK:              .type   __asan_check_store8_rn[[RN8]],@function
; CHECK-NEXT:         .weak   __asan_check_store8_rn[[RN8]]
; CHECK-NEXT:         .hidden __asan_check_store8_rn[[RN8]]
; CHECK-NEXT: __asan_check_store8_rn[[RN8]]:
; CHECK-NEXT:         movq    [[REG:.*]], %r8
; CHECK-NEXT:         shrq    $3, %r8
; CHECK-NEXT:         orq     $17592186044416, %r8{{.*}}
; CHECK-NEXT:         cmpb    $0, (%r8)
; CHECK-NEXT:         jne     [[FAIL:.*]]
; CHECK-NEXT: [[RET:.*]]:
; CHECK-NEXT:         retq
; CHECK-NEXT: [[FAIL]]:
; CHECK-NEXT:         movq    [[REG:.*]], %rdi
; CHECK-NEXT:         jmp     __asan_report_store8@PLT

; CHECK:              .type   __asan_check_store16_rn[[RN16]],@function
; CHECK-NEXT:         .weak   __asan_check_store16_rn[[RN16]]
; CHECK-NEXT:         .hidden __asan_check_store16_rn[[RN16]]
; CHECK-NEXT: __asan_check_store16_rn[[RN16]]:
; CHECK-NEXT:         movq    [[REG:.*]], %r8
; CHECK-NEXT:         shrq    $3, %r8
; CHECK-NEXT:         orq     $17592186044416, %r8{{.*}}
; CHECK-NEXT:         cmpw    $0, (%r8)
; CHECK-NEXT:         jne     [[FAIL:.*]]
; CHECK-NEXT: [[RET:.*]]:
; CHECK-NEXT:         retq
; CHECK-NEXT: [[FAIL]]:
; CHECK-NEXT:         movq    [[REG:.*]], %rdi
; CHECK-NEXT:         jmp     __asan_report_store16@PLT

declare void @llvm.asan.check.memaccess(i8*, i32 immarg)
