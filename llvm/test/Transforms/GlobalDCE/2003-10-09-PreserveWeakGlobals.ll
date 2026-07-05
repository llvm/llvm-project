; Weak variables should be preserved by global DCE!

; RUN: opt < %s -passes=globaldce -S | FileCheck %s

; CHECK: @A
@A = weak global i32 54
