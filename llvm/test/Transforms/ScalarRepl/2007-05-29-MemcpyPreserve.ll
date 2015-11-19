; RUN: opt < %s -scalarrepl -S | grep memcpy
; PR1421

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-apple-darwin8"

%struct.LongestMember = type { i8, i32 }
%struct.MyString = type { i32 }
%struct.UnionType = type { %struct.LongestMember }

define void @_Z4testP9UnionTypePS0_(%struct.UnionType* %p, %struct.UnionType** %pointerToUnion) {
entry:
  %tmp = alloca %struct.UnionType, align 8
  %tmp2 = getelementptr %struct.UnionType, %struct.UnionType* %tmp, i32 0, i32 0, i32 0
  %tmp13 = getelementptr %struct.UnionType, %struct.UnionType* %p, i32 0, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %tmp2, i8* %tmp13, i32 8, i1 false)
  %tmp5 = load %struct.UnionType*, %struct.UnionType** %pointerToUnion
  %tmp56 = getelementptr %struct.UnionType, %struct.UnionType* %tmp5, i32 0, i32 0, i32 0
  %tmp7 = getelementptr %struct.UnionType, %struct.UnionType* %tmp, i32 0, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %tmp56, i8* %tmp7, i32 8, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind
