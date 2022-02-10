; RUN: opt -passes='print<scalar-evolution>' < %s -disable-output 2>&1 | FileCheck %s

define i32 @basic(i32 %x, i32 %y) {
  %r = add i32 %x, %y
  ret i32 %r
}
