; RUN: llvm-as --opaque-pointers=0 < %s | llvm-dis --opaque-pointers=0 > %t1.ll
; RUN: FileCheck %s < %t1.ll
; RUN: llvm-as --opaque-pointers=0 < %t1.ll | llvm-dis --opaque-pointers=0 > %t2.ll
; RUN: diff %t1.ll %t2.ll
; RUN: opt --opaque-pointers=0 -O3 -S < %t1.ll | FileCheck %s

; CHECK: @i
@i = linkonce_odr global i32 1

; CHECK: f(){{.*}}prologue i32 1
define void @f() prologue i32 1 {
  ret void
}

; CHECK: g(){{.*}}prologue i32* @i
define void @g() prologue i32* @i {
  ret void
}
