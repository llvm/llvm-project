; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder < %s
; REQUIRES: x86-registered-target

define i32 @foo(<4 x float> %bar) nounwind {
entry:
; CHECK: call i32 @llvm.x86.sse41.ptestc(<2 x i64>
 %res1 = call i32 @llvm.x86.sse41.ptestc(<4 x float> %bar, <4 x float> %bar)
; CHECK: call i32 @llvm.x86.sse41.ptestz(<2 x i64> 
 %res2 = call i32 @llvm.x86.sse41.ptestz(<4 x float> %bar, <4 x float> %bar)
; CHECK: call i32 @llvm.x86.sse41.ptestnzc(<2 x i64>
 %res3 = call i32 @llvm.x86.sse41.ptestnzc(<4 x float> %bar, <4 x float> %bar)
 %add1 = add i32 %res1, %res2
 %add2 = add i32 %add1, %res2
 ret i32 %add2
}

; CHECK: declare i32 @llvm.x86.sse41.ptestc(<2 x i64>, <2 x i64>) #1
; CHECK: declare i32 @llvm.x86.sse41.ptestz(<2 x i64>, <2 x i64>) #1
; CHECK: declare i32 @llvm.x86.sse41.ptestnzc(<2 x i64>, <2 x i64>) #1

declare i32 @llvm.x86.sse41.ptestc(<4 x float>, <4 x float>) nounwind readnone
declare i32 @llvm.x86.sse41.ptestz(<4 x float>, <4 x float>) nounwind readnone
declare i32 @llvm.x86.sse41.ptestnzc(<4 x float>, <4 x float>) nounwind readnone

; CHECK: attributes #0 = { nounwind }
; CHECK: attributes #1 = { nocallback nofree nosync nounwind readnone willreturn }
