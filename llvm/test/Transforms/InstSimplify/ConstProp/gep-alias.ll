; RUN: opt -instcombine -S -o - %s | FileCheck %s
; Test that we don't replace an alias with its aliasee when simplifying GEPs.
; In this test case the transformation is invalid because it replaces the
; reference to the symbol "b" (which refers to whichever instance of "b"
; was chosen by the linker) with a reference to "a" (which refers to the
; specific instance of "b" in this module).

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = internal global [3 x i8*] zeroinitializer
@b = linkonce_odr alias [3 x i8*], [3 x i8*]* @a

define i8** @f() {
  ; CHECK: ret i8** getelementptr ([3 x i8*], [3 x i8*]* @b, i64 0, i64 1)
  ret i8** getelementptr ([3 x i8*], [3 x i8*]* @b, i64 0, i64 1)
}
