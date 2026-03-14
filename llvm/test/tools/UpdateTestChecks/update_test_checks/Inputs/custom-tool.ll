; RUN: llvm-extract -S --func=foo %s | FileCheck --check-prefixes=CHECK %s

define i32 @foo(i32 %x) {
  %y = add i32 %x, 1
  ret i32 %y
}
