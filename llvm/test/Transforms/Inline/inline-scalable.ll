; RUN: opt -S -passes=inline < %s | FileCheck %s

define void @func() {
; CHECK-LABEL: func
; CHECK-NEXT:    [[VEC_ADDR:%.*]] = alloca <vscale x 4 x i32>
; CHECK-NEXT:    call void @func()
; CHECK-NEXT:    ret void
  %vec.addr = alloca <vscale x 4 x i32>
  call void @func();
  ret void
}
