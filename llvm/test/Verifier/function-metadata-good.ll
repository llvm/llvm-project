; RUN: llvm-as < %s -o /dev/null 2>&1

; Function foo() is called 2,304 times at runtime.
define i32 @foo() !prof !0 {
  ret i32 0
}

!0 = !{!"function_entry_count", i32 2304}
