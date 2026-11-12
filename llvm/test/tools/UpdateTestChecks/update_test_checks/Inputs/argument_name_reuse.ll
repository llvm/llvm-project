; RUN: opt < %s -S | FileCheck %s

define i32 @reuse_arg_names(i32 %X, i32 %Y) {
  %Z = sub i32 %X, %Y
  %Q = add i32 %Z, %Y
  ret i32 %Q
}
