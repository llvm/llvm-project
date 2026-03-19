; RUN: llvm-as < %s > %t
; RUN: llvm-lto -list-symbols-only %t | FileCheck %s
; REQUIRES: default_triple

; CHECK-DAG: v    { data defined default }
; CHECK-DAG: va    { data defined default alias }
; CHECK-DAG: f    { function defined default }
; CHECK-DAG: fa    { function defined default alias }

@v = global i32 0
@va = alias i32, ptr @v
@fa = alias void (ptr), ptr @f

define void @f() {
  ret void
}
