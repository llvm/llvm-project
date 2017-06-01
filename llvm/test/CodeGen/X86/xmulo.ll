; RUN: llc %s -o - | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.8.0"

declare {i64, i1} @llvm.umul.with.overflow.i64(i64, i64) nounwind readnone
declare i32 @printf(i8*, ...)

@.str = private unnamed_addr constant [10 x i8] c"%llx, %d\0A\00", align 1

define i32 @t1() nounwind {
; CHECK-LABEL: t1:
; CHECK:  pushl $0
; CHECK:  pushl $0
; CHECK:  pushl $72

    %1 = call {i64, i1} @llvm.umul.with.overflow.i64(i64 9, i64 8)
    %2 = extractvalue {i64, i1} %1, 0
    %3 = extractvalue {i64, i1} %1, 1
    %4 = zext i1 %3 to i32
    %5 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str, i32 0, i32 0), i64 %2, i32 %4)
    ret i32 0
}

define i32 @t2() nounwind {
; CHECK-LABEL: t2:
; CHECK:  pushl $0
; CHECK:  pushl $0
; CHECK:  pushl $0

    %1 = call {i64, i1} @llvm.umul.with.overflow.i64(i64 9, i64 0)
    %2 = extractvalue {i64, i1} %1, 0
    %3 = extractvalue {i64, i1} %1, 1
    %4 = zext i1 %3 to i32
    %5 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str, i32 0, i32 0), i64 %2, i32 %4)
    ret i32 0
}

define i32 @t3() nounwind {
; CHECK-LABEL: t3:
; CHECK:  pushl $1
; CHECK:  pushl $-1
; CHECK:  pushl $-9

    %1 = call {i64, i1} @llvm.umul.with.overflow.i64(i64 9, i64 -1)
    %2 = extractvalue {i64, i1} %1, 0
    %3 = extractvalue {i64, i1} %1, 1
    %4 = zext i1 %3 to i32
    %5 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str, i32 0, i32 0), i64 %2, i32 %4)
    ret i32 0
}
