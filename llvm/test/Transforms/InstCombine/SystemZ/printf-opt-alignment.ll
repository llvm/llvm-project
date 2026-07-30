; RUN: opt < %s --passes=instcombine -S | FileCheck %s
;
; Check that string replacements inserted by the instcombiner are properly aligned.
; The specific case checked replaces `printf("foo\n")` with `puts("foo")`

target datalayout = "i8:8:16"

@msg1 = constant [17 x i8] c"Alignment Check\0A\00", align 2
; CHECK: c"Alignment Check\00", align 2

; Function Attrs: noinline nounwind
define dso_local void @foo() #0 {
  %call = call signext i32 (ptr, ...) @printf(ptr noundef @msg1)
  ret void
}

declare signext i32 @printf(ptr noundef, ...) #1
