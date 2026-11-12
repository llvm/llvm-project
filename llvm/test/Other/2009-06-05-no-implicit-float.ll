
; RUN: opt < %s -passes=verify -S | grep noimplicitfloat
declare void @f() noimplicitfloat

