; RUN: opt -passes='metarenamer<rename-exclude-function-prefixes=my_func;rename-exclude-function-prefixes=some_func>' -S %s | FileCheck %s

; CHECK: define i32 @my_func(
define i32 @my_func(i32, i32) {
  %3 = add i32 %0, %1
  ret i32 %3
}

; CHECK: define i32 @some_func(
define i32 @some_func(i32, i32) {
  %3 = add i32 %0, %1
  ret i32 %3
}

