; RUN: llc -march=xcore < %s | FileCheck %s

define i32 @switch(i32 %i) {
entry:
        switch i32 %i, label %default [
                 i32 0, label %bb0
                 i32 1, label %bb1
                 i32 2, label %bb2
                 i32 3, label %bb3
        ]
; CHECK-NOT: shl
; CHECK: bru
; CHECK: .jmptable
bb0:
        ret i32 0
bb1:
        ret i32 1
bb2:
        ret i32 2
bb3:
        ret i32 3
default:
        ret i32 4
}
