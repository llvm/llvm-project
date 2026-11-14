;; Check that a block ending in a return followed by SEH_EpilogEnd is not
;; considered to fall through into a following EH pad successor.

; RUN: llc -mtriple=thumbv7-windows -o - %s -verify-machineinstrs | FileCheck %s
; RUN: llc -mtriple=thumbv7-windows -filetype=obj -o %t.obj %s -verify-machineinstrs

; CHECK-LABEL: f:
; CHECK:              bl      g
; CHECK:       $MBB0_{{[0-9]+}}: @ %exit
; CHECK-NEXT:         .seh_startepilogue
; CHECK-NEXT:         pop     {r6, pc}
; CHECK-NEXT:         .seh_save_regs  {r6, lr}
; CHECK-NEXT:         .seh_endepilogue
; CHECK-NEXT:  $MBB0_{{[0-9]+}}: @ %__except
; CHECK-NEXT:  $ehgcr_0_{{[0-9]+}}:
; CHECK-NEXT:         .seh_startepilogue
; CHECK-NEXT:         pop.w   {r6, lr}
; CHECK-NEXT:         .seh_save_regs_w        {r6, lr}
; CHECK-NEXT:         b.w     h
; CHECK-NEXT:         .seh_nop_w
; CHECK-NEXT:         .seh_endepilogue

define void @f(ptr %p, i1 %c) personality ptr @__C_specific_handler {
entry:
  br i1 %c, label %store, label %try

try:
  invoke void @g()
          to label %try.cont unwind label %catch.dispatch

try.cont:
  invoke void @llvm.seh.try.end()
          to label %cleanup unwind label %catch.dispatch

catch.dispatch:
  %cs = catchswitch within none [label %__except] unwind to caller

__except:
  %cp = catchpad within %cs [ptr null]
  catchret from %cp to label %__except.ret

exit:
  ret void

__except.ret:
  tail call void @h()
  br label %exit

store:
  store ptr null, ptr %p, align 4
  br label %exit

cleanup:
  call void @llvm.lifetime.end.p0(ptr poison)
  br label %exit
}

declare i32 @__C_specific_handler(...)
declare void @g()
declare void @h()
