; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

%rt2 = type { i32, { i8, %rt2, i8 }, i32 }

define i32 @f(%rt2* %p) nounwind {
entry:
  ; Check that recursive types trigger an error instead of segfaulting, when
  ; the recursion isn't through a pointer to the type.
  ; CHECK: loading unsized types is not allowed
  %0 = load %rt2, %rt2* %p
  ret i32 %0
}
