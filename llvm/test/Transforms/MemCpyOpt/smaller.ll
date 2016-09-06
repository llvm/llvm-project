; RUN: opt -memcpyopt -S < %s | FileCheck %s
; RUN: opt -passes=memcpyopt -S < %s | FileCheck %s
; rdar://8875553

; Memcpyopt shouldn't optimize the second memcpy using the first
; because the first has a smaller size.

; CHECK: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %tmp, i8* getelementptr inbounds (%struct.s, %struct.s* @cell, i32 0, i32 0, i32 0), i32 16, i32 4, i1 false)

target datalayout = "e-p:32:32:32"

%struct.s = type { [11 x i8], i32 }

@.str = private constant [11 x i8] c"0123456789\00"
@cell = external global %struct.s

declare void @check(%struct.s* byval %p) nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind

define void @foo() nounwind {
entry:
  %agg.tmp = alloca %struct.s, align 4
  store i32 99, i32* getelementptr inbounds (%struct.s, %struct.s* @cell, i32 0, i32 1), align 4
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* getelementptr inbounds (%struct.s, %struct.s* @cell, i32 0, i32 0, i32 0), i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i32 0, i32 0), i32 11, i32 1, i1 false)
  %tmp = getelementptr inbounds %struct.s, %struct.s* %agg.tmp, i32 0, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %tmp, i8* getelementptr inbounds (%struct.s, %struct.s* @cell, i32 0, i32 0, i32 0), i32 16, i32 4, i1 false)
  call void @check(%struct.s* byval %agg.tmp)
  ret void
}
