
; RUN: opt < %s -verify -S | grep noimplicitfloat
declare void @f() noimplicitfloat

