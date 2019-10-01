; RUN: llc -O1 -mtriple=x86_64-unknown-linux-gnu -mcpu=core2 -relocation-model=pic -frame-pointer=all < %s | FileCheck %s
; <rdar://problem/8124405>

%struct.type = type { %struct.subtype*, i32, i8, i32, i8, i32, i32, i32, i32, i32, i8, i32, i32, i32, i32, i32, [256 x i32], i32, [257 x i32], [257 x i32], i32*, i16*, i8*, i32, i32, i32, i32, i32, [256 x i8], [16 x i8], [256 x i8], [4096 x i8], [16 x i32], [18002 x i8], [18002 x i8], [6 x [258 x i8]], [6 x [258 x i32]], [6 x [258 x i32]], [6 x [258 x i32]], [6 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32*, i32*, i32* }
%struct.subtype = type { i8*, i32, i32, i32, i8*, i32, i32, i32, i8*, i8* (i8*, i32, i32)*, void (i8*, i8*)*, i8* }

define i32 @func(%struct.type* %s) nounwind optsize ssp {
entry:
  %tmp1 = getelementptr inbounds %struct.type, %struct.type* %s, i32 0, i32 1
  %tmp2 = load i32, i32* %tmp1, align 8
  %tmp3 = icmp eq i32 %tmp2, 10
  %tmp4 = getelementptr inbounds %struct.type, %struct.type* %s, i32 0, i32 40
  br i1 %tmp3, label %bb, label %entry.bb1_crit_edge

entry.bb1_crit_edge:
  br label %bb1

bb:

; The point of this code is that %rdi is set to %rdi+64036 for the rep;stosl
; statement. It can be an ADD or LEA instruction, it's not important which one
; it is.
;
; CHECK: # %bb
; CHECK: leaq	64036(%rdx), %rdi
; CHECK: rep;stosl

  %tmp5 = bitcast i32* %tmp4 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 4 %tmp5, i8 0, i64 84, i1 false)
  %tmp6 = getelementptr inbounds %struct.type, %struct.type* %s, i32 0, i32 62
  store i32* null, i32** %tmp6, align 8
  br label %bb1

bb1:
  store i32 10, i32* %tmp1, align 8
  ret i32 42
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) nounwind
