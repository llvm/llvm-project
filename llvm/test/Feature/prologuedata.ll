; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: FileCheck %s < %t1.ll
; RUN: llvm-as < %t1.ll | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll
; RUN: opt -O3 -S < %t1.ll | FileCheck %s

; CHECK: @i
@i = linkonce_odr global i32 1

; CHECK: f(){{.*}}prologue i32 1
define void @f() prologue i32 1 {
  ret void
}

; CHECK: g(){{.*}}prologue ptr @i
define void @g() prologue ptr @i {
  ret void
}
