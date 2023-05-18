; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

%rt1 = type { i32, { i8, %rt2, i8 }, i32 }
%rt2 = type { i64, { i6, %rt3 } }
%rt3 = type { %rt1 }

define i32 @main() nounwind {
entry:
  ; Check that mutually recursive types trigger an error instead of segfaulting,
  ; when the recursion isn't through a pointer to the type.
  ; CHECK: Cannot allocate unsized type
  %0 = alloca %rt2
  ret i32 0
}
