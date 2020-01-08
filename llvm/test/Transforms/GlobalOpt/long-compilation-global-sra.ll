; RUN: opt %s --O0 -globalopt -S -o -

; This is a regression test against very slow execution...
; In bad case it should fail by timeout.

; Hand-reduced from this example.
; clang++ -mllvm -disable-llvm-optzns

;#include <stdio.h>
;
;namespace {
;  char LargeBuffer[64 * 1024 * 1024];
;}
;
;int main ( void ) {
;
;    LargeBuffer[0] = 0;
;
;    printf("");
;
;    return LargeBuffer[0] == 0;
;}

; check that global array LargeBufferE was optimized out
; and local variable LargeBufferE.0 was used instead.

; CHECK-NOT: global
; CHECK: main()
; CHECK-NEXT: LargeBufferE.0
; CHECK-NOT: global

; ModuleID = 'test.cpp'
source_filename = "test.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@LargeBufferE = internal global [67108864 x i8] zeroinitializer, align 16
@.str = private unnamed_addr constant [1 x i8] c"\00", align 1

; Function Attrs: norecurse uwtable
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  store i8 0, i8* getelementptr inbounds ([67108864 x i8], [67108864 x i8]* @LargeBufferE, i64 0, i64 0), align 16
  %2 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str, i64 0, i64 0))
  %3 = load i8, i8* getelementptr inbounds ([67108864 x i8], [67108864 x i8]* @LargeBufferE, i64 0, i64 0), align 16
  %4 = sext i8 %3 to i32
  %5 = icmp eq i32 %4, 0
  %6 = zext i1 %5 to i32
  ret i32 %6
}

declare dso_local i32 @printf(i8*, ...) #0

attributes #0 = { norecurse uwtable }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0 "}
