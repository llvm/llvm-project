; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

declare void @rt_init()

declare i32 @__CxxFrameHandler3(...)

define void @test1(ptr %fp, i64 %n) personality ptr @__CxxFrameHandler3 {
entry:
  %t.i = alloca ptr
  %t.ii = alloca i8
  %.alloca8 = alloca i8, i64 %n
  store volatile i8 0, ptr %t.ii
  store volatile i8 0, ptr %.alloca8
  invoke void @rt_init()
          to label %try.cont unwind label %catch.switch

try.cont:
  invoke void %fp()
          to label %exit unwind label %catch.switch

exit:
  ret void

catch.pad:
  %cp = catchpad within %cs [ptr null, i32 0, ptr %t.i]
  catchret from %cp to label %exit

catch.switch:
  %cs = catchswitch within none [label %catch.pad] unwind to caller
}

; CHECK-LABEL: test1:
; CHECK:      movabsq $15, %rax
; CHECK-NEXT: addq    %rdx, %rax
; CHECK-NEXT: andq    $-16, %rax
; CHECK-NEXT: callq   __chkstk
; CHECK-NEXT: subq    %rax, %rsp
; CHECK-NEXT: leaq    32(%rsp), %rax
; CHECK-NEXT: movb    $0, -9(%rbp)
; CHECK-NEXT: movb    $0, (%rax)
; CHECK:      callq   rt_init
; CHECK-NOT:  subq
; CHECK-NOT:  addq
; CHECK:      callq   *%rsi
; CHECK-LABEL: "?catch$3@?0?test1@4HA":
; CHECK:      leaq    48(%rdx), %rbp
; CHECK-LABEL: $handlerMap$0$test1:
; CHECK:      .long   0
; CHECK-NEXT: .long   0
; CHECK-NEXT: .long   48
; CHECK-NEXT: .long   "?catch$3@?0?test1@4HA"@IMGREL
; CHECK-NEXT: .long   72

define void @test2(ptr %fp, i64 %n) personality ptr @__CxxFrameHandler3 {
entry:
  %t.i = alloca i128
  %.alloca8 = alloca i8, i64 %n
  store volatile i8 0, ptr %.alloca8
  invoke void @rt_init()
          to label %try.cont unwind label %catch.switch

try.cont:
  invoke void %fp()
          to label %exit unwind label %catch.switch

exit:
  ret void

catch.pad:
  %cp = catchpad within %cs [ptr null, i32 0, ptr %t.i]
  catchret from %cp to label %exit

catch.switch:
  %cs = catchswitch within none [label %catch.pad] unwind to caller
}

; CHECK-LABEL: test2:
; CHECK:      movabsq $15, %rax
; CHECK-NEXT: addq    %rdx, %rax
; CHECK-NEXT: andq    $-16, %rax
; CHECK-NEXT: callq   __chkstk
; CHECK-NEXT: subq    %rax, %rsp
; CHECK-NEXT: leaq    32(%rsp), %rax
; CHECK-NEXT: movb    $0, (%rax)
; CHECK:      callq   rt_init
; CHECK-NOT:  subq
; CHECK-NOT:  addq
; CHECK:      callq   *%rsi
; CHECK-LABEL: "?catch$3@?0?test2@4HA":
; CHECK:      leaq    64(%rdx), %rbp
; CHECK-LABEL: $handlerMap$0$test2:
; CHECK:      .long   0
; CHECK-NEXT: .long   0
; CHECK-NEXT: .long   48
; CHECK-NEXT: .long   "?catch$3@?0?test2@4HA"@IMGREL
; CHECK-NEXT: .long   72
