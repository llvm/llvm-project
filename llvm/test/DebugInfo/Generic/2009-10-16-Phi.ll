; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

define i32 @foo() {
E:
   br label %B2
B1:
   br label %B2
B2:
; CHECK: error: invalid !dbg metadata
   %0 = phi i32 [ 0, %E ], [ 1, %B1 ], !dbg !0
   ret i32 %0
}

!0 = !{i32 42}
