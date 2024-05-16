; RUN: llc -verify-machineinstrs -mtriple=aarch64-pc-windows-msvc < %s \
; RUN:     | FileCheck --check-prefix=CHECK %s

; Loosely based on IR for this C++ source code:
;   void f(int p);
;   void try_in_catch() {
;     try {
;       f(1);
;     } catch (...) {
;       try {
;         f(2);
;       } catch (...) {
;         f(3);
;       }
;     }
;   }

declare void @f(i32 %p)
declare i32 @__CxxFrameHandler3(...)

define i32 @try_in_catch() personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @f(i32 1)
          to label %try.cont unwind label %catch.dispatch.1
try.cont:
  ret i32 0

catch.dispatch.1:
  %cs1 = catchswitch within none [label %handler1] unwind to caller
handler1:
  %h1 = catchpad within %cs1 [ptr null, i32 64, ptr null]
  invoke void @f(i32 2) [ "funclet"(token %h1) ]
          to label %catchret1 unwind label %catch.dispatch.2
catchret1:
  catchret from %h1 to label %try.cont

catch.dispatch.2:
  %cs2 = catchswitch within %h1 [label %handler2] unwind to caller
handler2:
  %h2 = catchpad within %cs2 [ptr null, i32 64, ptr null]
  call void @f(i32 3)
  catchret from %h2 to label %catchret1
}

; CHECK-LABEL: $cppxdata$try_in_catch:
; CHECK-NEXT: .word   429065506
; CHECK-NEXT: .word   4
; CHECK-NEXT: .word   ($stateUnwindMap$try_in_catch)
; CHECK-NEXT: .word   2
; CHECK-NEXT: .word   ($tryMap$try_in_catch)
; ip2state num + ptr
; CHECK-NEXT: .word   7
; CHECK-NEXT: .word   ($ip2state$try_in_catch)
; unwindhelp offset
; CHECK-NEXT: .word   -16
; CHECK-NEXT: .word   0
; EHFlags
; CHECK-NEXT: .word   1

; CHECK-LABEL: $tryMap$try_in_catch:
; CHECK-NEXT: .word   0
; CHECK-NEXT: .word   0
; CHECK-NEXT: .word   3
; CHECK-NEXT: .word   1
; CHECK-NEXT: .word   ($handlerMap$0$try_in_catch)
; CHECK-NEXT: .word   2
; CHECK-NEXT: .word   2
; CHECK-NEXT: .word   3
; CHECK-NEXT: .word   1
; CHECK-NEXT: .word   ($handlerMap$1$try_in_catch)

; CHECK: $handlerMap$0$try_in_catch:
; CHECK-NEXT:   .word   64
; CHECK-NEXT:   .word   0
; CHECK-NEXT:   .word   0
; CHECK-NEXT:   .word   "?catch${{[0-9]+}}@?0?try_in_catch@4HA"
; CHECK-NEXT:   .word   0

; CHECK: $handlerMap$1$try_in_catch:
; CHECK-NEXT:   .word   64
; CHECK-NEXT:   .word   0
; CHECK-NEXT:   .word   0
; CHECK-NEXT:   .word   "?catch${{[0-9]+}}@?0?try_in_catch@4HA"
; CHECK-NEXT:   .word   0

; CHECK: $ip2state$try_in_catch:
; CHECK-NEXT: .word   .Lfunc_begin0@IMGREL
; CHECK-NEXT: .word   -1
; CHECK-NEXT: .word   .Ltmp0@IMGREL
; CHECK-NEXT: .word   0
; CHECK-NEXT: .word   .Ltmp1@IMGREL
; CHECK-NEXT: .word   -1
; CHECK-NEXT: .word   "?catch$2@?0?try_in_catch@4HA"@IMGREL
; CHECK-NEXT: .word   1
; CHECK-NEXT: .word   .Ltmp2@IMGREL
; CHECK-NEXT: .word   2
; CHECK-NEXT: .word   .Ltmp3@IMGREL
; CHECK-NEXT: .word   1
; CHECK-NEXT: .word   "?catch$4@?0?try_in_catch@4HA"@IMGREL
; CHECK-NEXT: .word   3
