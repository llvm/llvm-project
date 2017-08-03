; RUN: llc < %s -march=x86-64 -O0 | FileCheck %s
; Make sure fast-isel doesn't screw up aggregate constants.
; (Failing out is okay, as long as we don't miscompile.)

%bar = type { i32 }

define i32 @foo()  {
  %tmp = extractvalue %bar { i32 3 }, 0
  ret i32 %tmp
; CHECK: movl $3, %eax
}
