; Test the complex GetElementPtr instruction handling in the EfficiencySanitizer
; cache fragmentation tool.
;
; RUN: opt < %s -esan -esan-cache-frag -S | FileCheck %s

; Code from http://llvm.org/docs/LangRef.html#getelementptr-instruction
; struct RT {
;   char A;
;   int B[10][20];
;   char C;
; };
; struct ST {
;   int X;
;   double Y;
;   struct RT Z;
; };
;
; int *foo(struct ST *s) {
;   return &s[1].Z.B[5][13];
; }

%struct.RT = type { i8, [10 x [20 x i32]], i8 }
%struct.ST = type { i32, double, %struct.RT }

define i32* @foo(%struct.ST* %s) nounwind uwtable readnone optsize ssp {
entry:
  %arrayidx = getelementptr inbounds %struct.ST, %struct.ST* %s, i64 1, i32 2, i32 1, i64 5, i64 13
  ret i32* %arrayidx
}

; CHECK:        %0 = load i64, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.ST#3#13#3#11", i32 0, i32 3)
; CHECK-NEXT:   %1 = add i64 %0, 1
; CHECK-NEXT:   store i64 %1, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.ST#3#13#3#11", i32 0, i32 3)
; CHECK-NEXT:   %2 = load i64, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.ST#3#13#3#11", i32 0, i32 2)
; CHECK-NEXT:   %3 = add i64 %2, 1
; CHECK-NEXT:   store i64 %3, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.ST#3#13#3#11", i32 0, i32 2)
; CHECK-NEXT:   %4 = load i64, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.RT#3#11#14#11", i32 0, i32 1)
; CHECK-NEXT:   %5 = add i64 %4, 1
; CHECK-NEXT:   store i64 %5, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.RT#3#11#14#11", i32 0, i32 1)
; CHECK-NEXT:   %arrayidx = getelementptr inbounds %struct.ST, %struct.ST* %s, i64 1, i32 2, i32 1, i64 5, i64 13
; CHECK-NEXT:   ret i32* %arrayidx
