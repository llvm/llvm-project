; RUN: opt < %s -passes=globaldce -S | FileCheck %s

; CHECK-NOT: global
@X = external global i32
@Y = internal global i32 7

