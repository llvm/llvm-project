; RUN: llc -disable-constant-hoisting < %s | FileCheck %s
target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

@d = global i32 8, align 4

define i32 @main() {
entry:
  %load = load i32, i32* @d, align 4
  %conv1 = zext i32 %load to i64
  %shl = shl i64 %conv1, 1
  %mul = and i64 %shl, 4294967312
  %cmp = icmp ugt i64 4294967295, %mul
  %zext = zext i1 %cmp to i32
  ret i32 %zext
}
; CHECK: main:
; CHECK:   movl    d, %[[load:.*]]
; CHECK:   movl    %[[load]], %[[copy:.*]]
; CHECK:   shrl    $31, %[[copy]]
; CHECK:   addl    %[[load]], %[[load]]
