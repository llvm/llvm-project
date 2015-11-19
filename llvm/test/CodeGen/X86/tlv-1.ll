; RUN: llc < %s -mtriple x86_64-apple-darwin -mcpu=core2 | FileCheck %s

%struct.A = type { [48 x i8], i32, i32, i32 }

@c = external thread_local global %struct.A, align 4

define void @main() nounwind ssp {
; CHECK-LABEL: main:
entry:
  call void @llvm.memset.p0i8.i64(i8* getelementptr inbounds (%struct.A, %struct.A* @c, i32 0, i32 0, i32 0), i8 0, i64 60, i32 1, i1 false)
  unreachable  
  ; CHECK: movq    _c@TLVP(%rip), %rdi
  ; CHECK-NEXT: callq   *(%rdi)
  ; CHECK-NEXT: movl    $0, 56(%rax)
  ; CHECK-NEXT: movq    $0, 48(%rax)
}

; rdar://10291355
define i32 @test() nounwind readonly ssp {
entry:
; CHECK-LABEL: test:
; CHECK: movq _a@TLVP(%rip),
; CHECK: callq *
; CHECK: movl (%rax), [[REGISTER:%[a-z]+]]
; CHECK: movq _b@TLVP(%rip),
; CHECK: callq *
; CHECK: subl (%rax), [[REGISTER]]
  %0 = load i32, i32* @a, align 4
  %1 = load i32, i32* @b, align 4
  %sub = sub nsw i32 %0, %1
  ret i32 %sub
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) nounwind

@a = thread_local global i32 0                    ; <i32*> [#uses=0]
@b = thread_local global i32 0                    ; <i32*> [#uses=0]

; CHECK: .tbss _a$tlv$init, 4, 2
; CHECK:        .section        __DATA,__thread_vars,thread_local_variables
; CHECK:        .globl  _a
; CHECK: _a:
; CHECK:        .quad   __tlv_bootstrap
; CHECK:        .quad   0
; CHECK:        .quad   _a$tlv$init

; CHECK: .tbss _b$tlv$init, 4, 2
; CHECK:        .globl  _b
; CHECK: _b:
; CHECK:        .quad   __tlv_bootstrap
; CHECK:        .quad   0
; CHECK:        .quad   _b$tlv$init
