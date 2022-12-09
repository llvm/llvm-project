
; RUN: opt -temporarily-allow-old-pass-syntax < %s -verify -S | grep noimplicitfloat
declare void @f() noimplicitfloat

