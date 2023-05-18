; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

define i32 @bad1() !prof !0 {
  ret i32 0
}

!0 = !{i32 123, i32 3}
; CHECK: assembly parsed, but does not verify as correct!
; CHECK-NEXT: expected string with name of the !prof annotation
; CHECK-NEXT: !0 = !{i32 123, i32 3}

define i32 @bad2() !prof !1 {
  ret i32 0
}

!1 = !{!"function_entry_count"}
; CHECK-NEXT: !prof annotations should have no less than 2 operands
; CHECK-NEXT: !1 = !{!"function_entry_count"}


define i32 @bad3() !prof !2 {
  ret i32 0
}

!2 = !{!"some_other_count", i64 200}
; CHECK-NEXT: first operand should be 'function_entry_count'
; CHECK-NEXT: !2 = !{!"some_other_count", i64 200}

define i32 @bad4() !prof !3 {
  ret i32 0
}

!3 = !{!"function_entry_count", !"string"}
; CHECK-NEXT: expected integer argument to function_entry_count
; CHECK-NEXT: !3 = !{!"function_entry_count", !"string"}
