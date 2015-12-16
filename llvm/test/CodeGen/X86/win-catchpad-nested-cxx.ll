; RUN: llc -verify-machineinstrs -mtriple=i686-pc-windows-msvc < %s \
; RUN:     | FileCheck --check-prefix=CHECK --check-prefix=X86 %s
; RUN: llc -verify-machineinstrs -mtriple=x86_64-pc-windows-msvc < %s \
; RUN:     | FileCheck --check-prefix=CHECK --check-prefix=X64 %s

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

define i32 @try_in_catch() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f(i32 1)
          to label %try.cont unwind label %catch.dispatch.1
try.cont:
  ret i32 0

catch.dispatch.1:
  %cs1 = catchswitch within none [label %handler1] unwind to caller
handler1:
  %h1 = catchpad within %cs1 [i8* null, i32 64, i8* null]
  invoke void @f(i32 2) [ "funclet"(token %h1) ]
          to label %catchret1 unwind label %catch.dispatch.2
catchret1:
  catchret from %h1 to label %try.cont

catch.dispatch.2:
  %cs2 = catchswitch within %h1 [label %handler2] unwind to caller
handler2:
  %h2 = catchpad within %cs2 [i8* null, i32 64, i8* null]
  call void @f(i32 3)
  catchret from %h2 to label %catchret1
}

; X86-LABEL: L__ehtable$try_in_catch:
; X64-LABEL: $cppxdata$try_in_catch:
; CHECK-NEXT: .long   429065506
; CHECK-NEXT: .long   4
; CHECK-NEXT: .long   ($stateUnwindMap$try_in_catch)
; CHECK-NEXT: .long   2
; CHECK-NEXT: .long   ($tryMap$try_in_catch)
; ip2state num + ptr
; X86-NEXT: .long   0
; X86-NEXT: .long   0
; X64-NEXT: .long   7
; X64-NEXT: .long   ($ip2state$try_in_catch)
; unwindhelp offset
; X64-NEXT: .long   40
; CHECK-NEXT: .long   0
; EHFlags
; CHECK-NEXT: .long   1

; CHECK: $tryMap$try_in_catch:
; CHECK-NEXT: .long   2
; CHECK-NEXT: .long   2
; CHECK-NEXT: .long   3
; CHECK-NEXT: .long   1
; CHECK-NEXT: .long   ($handlerMap$0$try_in_catch)
; CHECK-NEXT: .long   0
; CHECK-NEXT: .long   0
; CHECK-NEXT: .long   3
; CHECK-NEXT: .long   1
; CHECK-NEXT: .long   ($handlerMap$1$try_in_catch)

; CHECK: $handlerMap$0$try_in_catch:
; CHECK-NEXT:   .long   64
; CHECK-NEXT:   .long   0
; CHECK-NEXT:   .long   0
; CHECK-NEXT:   .long   "?catch${{[0-9]+}}@?0?try_in_catch@4HA"
; X64-NEXT:   .long   56

; CHECK: $handlerMap$1$try_in_catch:
; CHECK-NEXT:   .long   64
; CHECK-NEXT:   .long   0
; CHECK-NEXT:   .long   0
; CHECK-NEXT:   .long   "?catch${{[0-9]+}}@?0?try_in_catch@4HA"
; X64-NEXT:   .long   56

; X64: $ip2state$try_in_catch:
; X64-NEXT: .long   .Lfunc_begin0@IMGREL
; X64-NEXT: .long   -1
; X64-NEXT: .long   .Ltmp0@IMGREL+1
; X64-NEXT: .long   0
; X64-NEXT: .long   .Ltmp1@IMGREL+1
; X64-NEXT: .long   -1
; X64-NEXT: .long   "?catch$2@?0?try_in_catch@4HA"@IMGREL
; X64-NEXT: .long   1
; X64-NEXT: .long   .Ltmp2@IMGREL+1
; X64-NEXT: .long   2
; X64-NEXT: .long   .Ltmp3@IMGREL+1
; X64-NEXT: .long   1
; X64-NEXT: .long   "?catch$4@?0?try_in_catch@4HA"@IMGREL
; X64-NEXT: .long   3
