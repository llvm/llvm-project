; RUN: opt -O0 -S < %s | FileCheck %s
; RUN: opt -O1 -S < %s | FileCheck %s
; RUN: opt -O2 -S < %s | FileCheck %s
; RUN: opt -O3 -S < %s | FileCheck %s
; RUN: opt -Os -S < %s | FileCheck %s
; RUN: opt -Oz -S < %s | FileCheck %s

target datalayout = "e-p:64:64"

; Make sure that optimizations do not optimize inrange GEP.

@vtable = constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* null, i8* null] }

define void @foo(i8*** %p) {
  ;CHECK: store i8** getelementptr {{.*}} ({ [3 x i8*] }, { [3 x i8*] }* @vtable, i{{.*}} 0, inrange i32 0, i{{.*}} 3), i8*** %p
  store i8** getelementptr ({ [3 x i8*] }, { [3 x i8*] }* @vtable, i32 0, inrange i32 0, i32 3), i8*** %p
  ret void
}
