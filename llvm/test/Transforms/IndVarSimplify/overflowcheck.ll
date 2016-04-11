; RUN: opt < %s -indvars -liv-reduce -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

; CHECK-LABEL: @addwithoverflow
; CHECK-LABEL: loop1:
; CHECK-NOT: zext
; CHECK: add nsw
; CHECK: @llvm.sadd.with.overflow
; CHECK-LABEL: loop2:
; CHECK-NOT: extractvalue
; CHECK: add nuw
; CHECK: @llvm.sadd.with.overflow
; CHECK-LABEL: loop3:
; CHECK-NOT: extractvalue
; CHECK: ret
define i64 @addwithoverflow(i32 %n, i64* %a) {
entry:
  br label %loop0

loop0:
  %i = phi i32 [ 0, %entry ], [ %i1val, %loop3 ]
  %s = phi i32 [ 0, %entry ], [ %addsval, %loop3 ]
  %bc = icmp ult i32 %i, %n
  br i1 %bc, label %loop1, label %exit

loop1:
  %zxt = zext i32 %i to i64
  %ofs = shl nuw nsw i64 %zxt, 3
  %gep = getelementptr i64, i64* %a, i64 %zxt
  %v = load i64, i64* %gep, align 8
  %truncv = trunc i64 %v to i32
  %adds = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %s, i32 %truncv)
  %ovflows = extractvalue { i32, i1 } %adds, 1
  br i1 %ovflows, label %exit, label %loop2

loop2:
  %addsval = extractvalue { i32, i1 } %adds, 0
  %i1 = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %i, i32 1)
  %i1check = extractvalue { i32, i1 } %i1, 1
  br i1 %i1check, label %exit, label %loop3

loop3:
  %i1val = extractvalue { i32, i1 } %i1, 0
  %test = icmp slt i32 %i1val, %n
  br i1 %test, label %return, label %loop0

return:
  %ret = zext i32 %addsval to i64
  ret i64 %ret

exit:
  unreachable
}

declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32)
