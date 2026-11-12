; RUN: llc -mtriple=x86_64-apple-darwin -mattr=-sse < %s
; PR 7135

@x = common global i64 0                          ; <ptr> [#uses=1]

define i32 @foo() nounwind readonly ssp {
entry:
  %0 = load i64, ptr @x, align 8                      ; <i64> [#uses=1]
  %1 = uitofp i64 %0 to double                    ; <double> [#uses=1]
  %2 = fptosi double %1 to i32                    ; <i32> [#uses=1]
  ret i32 %2
}
