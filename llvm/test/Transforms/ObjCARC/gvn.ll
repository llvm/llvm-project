; RUN: opt -S -basicaa -objc-arc -gvn < %s | FileCheck %s

@x = common global i8* null, align 8

declare i8* @objc_retain(i8*)

; GVN should be able to eliminate this redundant load, with ARC-specific
; alias analysis.

; CHECK: @foo
; CHECK-NEXT: entry:
; CHECK-NEXT: %s = load i8** @x
; CHECK-NOT: load
; CHECK: ret i8* %s
define i8* @foo(i32 %n) nounwind {
entry:
  %s = load i8** @x
  %0 = tail call i8* @objc_retain(i8* %s) nounwind
  %t = load i8** @x
  ret i8* %s
}
