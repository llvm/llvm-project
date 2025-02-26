; RUN: llc -mtriple=x86_64-windows-msvc < %s | FileCheck %s

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define void @"\01?f@@YAXXZ"(i1 %B) personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @g()
          to label %unreachable unwind label %cleanupblock

cleanupblock:
  %cleanp = cleanuppad within none []
  call void @g() [ "funclet"(token %cleanp) ]
  cleanupret from %cleanp unwind label %catch.dispatch

catch.dispatch:
  %cs1 = catchswitch within none [label %catch] unwind to caller

catch:
  %cp = catchpad within %cs1 [ptr null, i32 64, ptr null]
  call void @g() [ "funclet"(token %cp) ]
  catchret from %cp to label %try.cont

try.cont:
  ret void

unreachable:
  unreachable
}


declare void @g()

declare i32 @__CxxFrameHandler3(...)

; Destructors need CFI but they shouldn't use the .seh_handler directive.
; CHECK: "?dtor$[[cleanup:[0-9]+]]@?0??f@@YAXXZ@4HA":
; CHECK: .seh_proc "?dtor$[[cleanup]]@?0??f@@YAXXZ@4HA"
; CHECK-NOT: .seh_handler __CxxFrameHandler3
; CHECK: LBB0_[[cleanup]]: # %cleanupblock{{$}}

; Emit CFI for pushing RBP.
; CHECK: movq    %rdx, 16(%rsp)
; CHECK: pushq   %rbp
; CHECK: .seh_pushreg %rbp

; Emit CFI for allocating from the stack pointer.
; CHECK: subq    $32, %rsp
; CHECK: .seh_stackalloc 32

; CHECK: leaq    48(%rdx), %rbp
; CHECK-NOT: .seh_setframe

; Prologue is done, emit the .seh_endprologue directive.
; CHECK: .seh_endprologue

; Make sure there is a nop after a call if the call precedes the epilogue.
; CHECK: callq g
; CHECK-NEXT: nop
; CHECK-NEXT: .seh_startepilogue

; Don't emit a reference to the LSDA.
; CHECK: .seh_handlerdata
; CHECK-NOT:  .long   ("$cppxdata$?f@@YAXXZ")@IMGREL
; CHECK-NEXT: .text
; CHECK: .seh_endproc

; CHECK: "?catch$[[catch:[0-9]+]]@?0??f@@YAXXZ@4HA":
; CHECK: .seh_proc "?catch$[[catch]]@?0??f@@YAXXZ@4HA"
; CHECK-NEXT: .seh_handler __CxxFrameHandler3, @unwind, @except
; CHECK: LBB0_[[catch]]: # %catch{{$}}

; Emit CFI for pushing RBP.
; CHECK: movq    %rdx, 16(%rsp)
; CHECK: pushq   %rbp
; CHECK: .seh_pushreg %rbp

; Emit CFI for allocating from the stack pointer.
; CHECK: subq    $32, %rsp
; CHECK: .seh_stackalloc 32

; CHECK: leaq    48(%rdx), %rbp
; CHECK-NOT: .seh_setframe

; Prologue is done, emit the .seh_endprologue directive.
; CHECK: .seh_endprologue

; Make sure there is at least one instruction after a call before the epilogue.
; CHECK: callq g
; CHECK-NEXT: leaq    .LBB0_{{[0-9]+}}(%rip), %rax
; CHECK-NEXT: .seh_startepilogue

; Emit a reference to the LSDA.
; CHECK: .seh_handlerdata
; CHECK-NEXT:  .long   ("$cppxdata$?f@@YAXXZ")@IMGREL
; CHECK-NEXT: .text
; CHECK: .seh_endproc
