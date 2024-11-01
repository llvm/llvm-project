; RUN: opt -O0 -S < %s | FileCheck %s
; RUN: opt -O1 -S < %s | FileCheck %s
; RUN: opt -O2 -S < %s | FileCheck %s
; RUN: opt -O3 -S < %s | FileCheck %s
; RUN: opt -Os -S < %s | FileCheck %s
; RUN: opt -Oz -S < %s | FileCheck %s

target datalayout = "e-p:64:64"

; Make sure that optimizations do not optimize inrange GEP.

@vtable = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr null] }

define void @foo(ptr %p) {
  ;CHECK: store ptr getelementptr {{.*}} ({ [3 x ptr] }, ptr @vtable, i{{.*}} 0, inrange i32 0, i{{.*}} 3), ptr %p
  store ptr getelementptr ({ [3 x ptr] }, ptr @vtable, i32 0, inrange i32 0, i32 3), ptr %p
  ret void
}
