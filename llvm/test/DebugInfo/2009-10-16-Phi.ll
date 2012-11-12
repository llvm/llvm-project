; RUN: llvm-as %s -disable-output

define i32 @foo() {
E:
   br label %B2
B1:
   br label %B2
B2:
   %0 = phi i32 [ 0, %E ], [ 1, %B1 ], !dbg !0
   ret i32 %0
}

!0 = metadata !{i32 42}
