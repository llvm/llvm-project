; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

%rt2 = type { i32, { i8, %rt2, i8 }, i32 }

define void @f(%rt2 %r, %rt2 *%p) nounwind {
entry:
  ; Check that recursive types trigger an error instead of segfaulting, when
  ; the recursion isn't through a pointer to the type.
  ; CHECK: storing unsized types is not allowed
  store %rt2 %r, %rt2 *%p
  ret void
}
