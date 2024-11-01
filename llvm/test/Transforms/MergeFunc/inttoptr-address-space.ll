; RUN: opt -passes=mergefunc -S < %s | FileCheck %s
target datalayout = "e-p:32:32:32-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-n8:16:32-S128"

%.qux.2496 = type { i32, %.qux.2497 }
%.qux.2497 = type { i8, i32 }
%.qux.2585 = type { i32, i32, i8* }

@g2 = external addrspace(1) constant [9 x i8], align 1
@g3 = internal unnamed_addr constant [1 x i8*] [i8* bitcast (i8* (%.qux.2585 addrspace(1)*)* @func35 to i8*)]


define internal i32 @func10(%.qux.2496 addrspace(1)* nocapture %this) align 2 {
bb:
  %tmp = getelementptr inbounds %.qux.2496, %.qux.2496 addrspace(1)* %this, i32 0, i32 1, i32 1
  %tmp1 = load i32, i32 addrspace(1)* %tmp, align 4
  ret i32 %tmp1
}

; Check for pointer bitwidth equal assertion failure
define internal i8* @func35(%.qux.2585 addrspace(1)* nocapture %this) align 2 {
bb:
; CHECK-LABEL: @func35(
; CHECK: %[[V2:.+]] = bitcast %.qux.2585 addrspace(1)* %{{.*}} to %.qux.2496 addrspace(1)*
; CHECK: %[[V3:.+]] = tail call i32 @func10(%.qux.2496 addrspace(1)* nocapture %[[V2]])
; CHECK: %{{.*}} = inttoptr i32 %[[V3]] to i8*
  %tmp = getelementptr inbounds %.qux.2585, %.qux.2585 addrspace(1)* %this, i32 0, i32 2
  %tmp1 = load i8*, i8* addrspace(1)* %tmp, align 4
  ret i8* %tmp1
}
