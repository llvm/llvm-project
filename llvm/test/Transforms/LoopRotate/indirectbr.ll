; RUN: opt < %s -S -loop-rotate -o - -verify-loop-info -verify-dom-info -verify-memoryssa | FileCheck %s

; PR5502
define void @z80_do_opcodes() nounwind {
entry:
  br label %while.cond

while.cond:                                       ; preds = %end_opcode, %entry
  br label %while.body

while.body:                                       ; preds = %while.cond
  br label %indirectgoto

run_opcode:                                       ; preds = %indirectgoto
  %tmp276 = load i8, i8* undef                        ; <i8> [#uses=1]
  br label %indirectgoto

if.else295:                                       ; preds = %divide_late
  br label %end_opcode

end_opcode:                                       ; preds = %indirectgoto, %sw.default42406, %sw.default, %if.else295
  %opcode.2 = phi i8 [ %opcode.0, %indirectgoto ], [ 0, %sw.default42406 ], [ undef, %sw.default ], [ %opcode.0, %if.else295 ] ; <i8> [#uses=0]
  switch i32 undef, label %while.cond [
    i32 221, label %sw.bb11691
    i32 253, label %sw.bb30351
  ]

sw.bb11691:                                       ; preds = %end_opcode
  br label %sw.default

sw.default:                                       ; preds = %sw.bb11691
  br label %end_opcode

sw.bb30351:                                       ; preds = %end_opcode
  br label %sw.default42406

sw.default42406:                                  ; preds = %sw.bb30351
  br label %end_opcode

indirectgoto:                                     ; preds = %run_opcode, %while.body
  %opcode.0 = phi i8 [ undef, %while.body ], [ %tmp276, %run_opcode ] ; <i8> [#uses=2]
  indirectbr i8* undef, [label %run_opcode, label %if.else295, label %end_opcode]
}

; CHECK-LABEL: @foo
define void @foo(i1 %a, i1 %b, i8* %c) {
; CHECK: entry
; CHECK-NEXT: br i1 %a, label %return, label %preheader
entry:
  br i1 %a, label %return, label %preheader

; CHECK: preheader:
; CHECK-NEXT:  br label %header
preheader:
  br label %header

; CHECK: header:
; CHECK-NEXT:  br i1 %b, label %return, label %body
header:
  br i1 %b, label %return, label %body

; CHECK: body:
; CHECK-NEXT:  indirectbr i8* %c, [label %return, label %latch]
body:
  indirectbr i8* %c, [label %return, label %latch]

; CHECK: latch:
; CHECK-NEXT:  br label %header
latch:
  br label %header

return:
  ret void
}
