; RUN: opt -passes='metarenamer<rename-exclude-function-prefixes="my_fu;nc">' -S %s | FileCheck %s

; CHECK: define i32 @"my_fu;nc"(
define i32 @"my_fu;nc"(i32, i32) {
  %3 = add i32 %0, %1
  ret i32 %3
}

