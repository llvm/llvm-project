; RUN: llvm-as %s -o /dev/null 2>&1

%rt2 = type { i32, { i8, ptr, i8 }, i32 }

define i32 @main() nounwind {
entry:
  ; Check that linked-list-style recursive types where the recursion is through
  ; a pointer of the type is valid for an alloca.
  %0 = alloca %rt2
  ret i32 0
}
