; RUN: llc -verify-machineinstrs < %s | FileCheck %s

; We used to have an issue where we inserted an MBB between invoke.cont.3 and
; its fallthrough target of ret void.

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc18.0.0"

@some_global = global i32 0

declare i32 @__CxxFrameHandler3(...)

declare void @g()

define void @f() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @g()
          to label %invoke.cont.3 unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchpad [i8* null, i32 64, i8* null]
          to label %catch unwind label %catchendblock

catch:                                            ; preds = %catch.dispatch
  catchret %0 to label %nrvo.skipdtor

catchendblock:                                    ; preds = %catch, %catch.dispatch
  catchendpad unwind to caller

invoke.cont.3:                                    ; preds = %entry
  store i32 123, i32* @some_global
  br label %nrvo.skipdtor

nrvo.skipdtor:                                    ; preds = %invoke.cont.3, %invoke.cont.4
  ret void
}

; CHECK-LABEL: _f: # @f
; CHECK: calll _g
; CHECK: movl $123, _some_global
; CHECK-NOT: jmp
; CHECK-NOT: movl {{.*}}, %esp
; CHECK: retl
; CHECK: addl $12, %ebp
; CHECK: jmp LBB0_{{.*}}
