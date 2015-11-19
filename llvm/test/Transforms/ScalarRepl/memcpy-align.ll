; RUN: opt -scalarrepl -S < %s | FileCheck %s
; PR6832
target datalayout =
"e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n32"
target triple = "arm-u-u"

%0 = type { %struct.anon, %struct.anon }
%struct.anon = type { [4 x i8] }

@c = external global %0                           ; <%0*> [#uses=1]

define void @good() nounwind {
entry:
  %x0 = alloca %struct.anon, align 4              ; <%struct.anon*> [#uses=2]
  %tmp = bitcast %struct.anon* %x0 to i8*         ; <i8*> [#uses=1]
  call void @llvm.memset.p0i8.i32(i8* %tmp, i8 0, i32 4, i1 false)
  %tmp1 = bitcast %struct.anon* %x0 to i8*        ; <i8*> [#uses=1]
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 getelementptr inbounds (%0, %0* @c, i32
0, i32 0, i32 0, i32 0), i8* %tmp1, i32 4, i32 4, i1 false)
  ret void
  
; CHECK: store i8 0, i8*{{.*}}, align 4
; CHECK: store i8 0, i8*{{.*}}, align 1
; CHECK: store i8 0, i8*{{.*}}, align 2
; CHECK: store i8 0, i8*{{.*}}, align 1
}

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i1) nounwind

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32,
i1) nounwind

