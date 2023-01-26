; Test to ensure no inlining is allowed into a caller with fewer nobuiltin attributes.
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -S -passes=inline | FileCheck %s
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -S -passes='cgscc(inline)' | FileCheck %s

; Make sure we don't inline callees into a caller with a superset of the
; no builtin attributes when -inline-caller-superset-nobuiltin=false.
; RUN: opt < %s -inline-caller-superset-nobuiltin=false -mtriple=x86_64-unknown-linux-gnu -S -passes='cgscc(inline)' | FileCheck %s --check-prefix=NOSUPERSET

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @allbuiltins() {
entry:
  %call = call i32 (...) @externalfunc()
  ret i32 %call
; CHECK-LABEL: allbuiltins
; CHECK: call i32 (...) @externalfunc()
}
declare i32 @externalfunc(...)

; We can inline a function that allows all builtins into one with a single
; nobuiltin.
define i32 @nobuiltinmemcpy() #0 {
entry:
  %call = call i32 @allbuiltins()
  ret i32 %call
; CHECK-LABEL: nobuiltinmemcpy
; CHECK-NOT: call i32 @allbuiltins()
; NOSUPERSET-LABEL: nobuiltinmemcpy
; NOSUPERSET: call i32 @allbuiltins()
}

; We can inline a function that allows all builtins into one with all
; nobuiltins.
define i32 @nobuiltins() #1 {
entry:
  %call = call i32 @allbuiltins()
  ret i32 %call
; CHECK-LABEL: nobuiltins
; CHECK-NOT: call i32 @allbuiltins()
; NOSUPERSET-LABEL: nobuiltins
; NOSUPERSET: call i32 @allbuiltins()
}

; We can inline a function with a single nobuiltin into one with all nobuiltins.
define i32 @nobuiltins2() #1 {
entry:
  %call = call i32 @nobuiltinmemcpy()
  ret i32 %call
; CHECK-LABEL: nobuiltins2
; CHECK-NOT: call i32 @nobuiltinmemcpy()
; NOSUPERSET-LABEL: nobuiltins2
; NOSUPERSET: call i32 @nobuiltinmemcpy()
}

; We can't inline a function with any given nobuiltin into one that allows all
; builtins.
define i32 @allbuiltins2() {
entry:
  %call = call i32 @nobuiltinmemcpy()
  ret i32 %call
; CHECK-LABEL: allbuiltins2
; CHECK: call i32 @nobuiltinmemcpy()
; NOSUPERSET-LABEL: allbuiltins2
; NOSUPERSET: call i32 @nobuiltinmemcpy()
}

; We can't inline a function with all nobuiltins into one that allows all
; builtins.
define i32 @allbuiltins3() {
entry:
  %call = call i32 @nobuiltins()
  ret i32 %call
; CHECK-LABEL: allbuiltins3
; CHECK: call i32 @nobuiltins()
; NOSUPERSET-LABEL: allbuiltins3
; NOSUPERSET: call i32 @nobuiltins()
}

; We can't inline a function with a specific nobuiltin into one with a
; different specific nobuiltin.
define i32 @nobuiltinmemset() #2 {
entry:
  %call = call i32 @nobuiltinmemcpy()
  ret i32 %call
; CHECK-LABEL: nobuiltinmemset
; CHECK: call i32 @nobuiltinmemcpy()
; NOSUPERSET-LABEL: nobuiltinmemset
; NOSUPERSET: call i32 @nobuiltinmemcpy()
}

attributes #0 = { "no-builtin-memcpy" }
attributes #1 = { "no-builtins" }
attributes #2 = { "no-builtin-memset" }
