; Test that the memset library call simplifier works correctly.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

declare i8* @memset(i8*, i32, i32)

; Check memset(mem1, val, size) -> llvm.memset(mem1, val, size, 1).

define i8* @test_simplify1(i8* %mem, i32 %val, i32 %size) {
; CHECK-LABEL: @test_simplify1(
  %ret = call i8* @memset(i8* %mem, i32 %val, i32 %size)
; CHECK: call void @llvm.memset
  ret i8* %ret
; CHECK: ret i8* %mem
}

; FIXME: memset(malloc(x), 0, x) -> calloc(1, x)

define float* @pr25892(i32 %size) #0 {
entry:
  %call = tail call i8* @malloc(i32 %size) #1
  %cmp = icmp eq i8* %call, null
  br i1 %cmp, label %cleanup, label %if.end
if.end:
  %bc = bitcast i8* %call to float*
  %call2 = tail call i8* @memset(i8* nonnull %call, i32 0, i32 %size) #1
  br label %cleanup
cleanup:
  %retval.0 = phi float* [ %bc, %if.end ], [ null, %entry ]
  ret float* %retval.0

; CHECK-LABEL: @pr25892(
; CHECK:       entry:
; CHECK-NEXT:    %call = tail call i8* @malloc(i32 %size) #1
; CHECK-NEXT:    %cmp = icmp eq i8* %call, null
; CHECK-NEXT:    br i1 %cmp, label %cleanup, label %if.end
; CHECK:       if.end: 
; CHECK-NEXT:    %bc = bitcast i8* %call to float*
; CHECK-NEXT:    call void @llvm.memset.p0i8.i32(i8* nonnull %call, i8 0, i32 %size, i32 1, i1 false)
; CHECK-NEXT:    br label %cleanup
; CHECK:       cleanup:
; CHECK-NEXT:    %retval.0 = phi float* [ %bc, %if.end ], [ null, %entry ]
; CHECK-NEXT:    ret float* %retval.0
}

declare noalias i8* @malloc(i32) #1
declare i64 @llvm.objectsize.i64.p0i8(i8*, i1) #2

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind }
attributes #2 = { nounwind readnone }

