; RUN: llc < %s -mtriple=x86_64-pc-linux-gnu | FileCheck %s
@"foo\22bar" = global i32 42
; CHECK: .globl "foo\"bar"

@"foo\0abar" = global i32 42
; CHECK: .globl "foo\nbar"
