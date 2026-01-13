; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 -pass-remarks='loop-interchange' -pass-remarks-missed='loop-interchange' -pass-remarks-output=%t -disable-output -S
; RUN: FileCheck --input-file=%t %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"

; This is a reduced test case for the example in  "large-nested-6d.ll". For a
; full description of the purpose this test and its complexities, see that file.
;
; This reproducer contains the perfectly nested sub part of that bigger loop
; nest:
;
;        for i=1 to NX
;         for j=1 to NY
;          for IL=1 to NX
;           load GlobC(i,IL,L)
;           load GlobG(i,IL,L)
;           load GlobE(i,IL,L)
;           load GlobI(i,IL,L)
;           for JL=1 to NY
;            load GlobD(j,JL,M)
;            load GlobH(j,JL,M)
;            load GlobF(j,JL,M)
;            load GlobJ(j,JL,M)
;            store GlobL(NY*i+j,NY*IL+JL)
;           End
;          End
;         End
;        End
;
; This reproducer is useful to focus on only on the 2nd challenge: the data
; dependence analysis problem, and not worry about the rest of loop nest
; structure.
;
; TODO:
;
; If loop-interchange is able to deal with imperfectly nested loops, this
; test is redundant and we only need to keep "large-nested-6d.ll".
;
; CHECK:        --- !Analysis
; CHECK-NEXT:   Pass:            loop-interchange
; CHECK-NEXT:   Name:            Dependence
; CHECK-NEXT:   Function:        test
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - String:          Computed dependence info, invoking the transform.
; CHECK-NEXT:   ...
; CHECK-NEXT:   --- !Missed
; CHECK-NEXT:   Pass:            loop-interchange
; CHECK-NEXT:   Name:            Dependence
; CHECK-NEXT:   Function:        test
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - String:          All loops have dependencies in all directions.
; CHECK-NEXT:   ...

@GlobC = local_unnamed_addr global [54 x [54 x [54 x double]]] zeroinitializer
@GlobD = local_unnamed_addr global [54 x [54 x [54 x double]]] zeroinitializer
@GlobE = local_unnamed_addr global [54 x [54 x [54 x double]]] zeroinitializer
@GlobF = local_unnamed_addr global [54 x [54 x [54 x double]]] zeroinitializer
@GlobG = local_unnamed_addr global [54 x [54 x [54 x double]]] zeroinitializer
@GlobH = local_unnamed_addr global [54 x [54 x [54 x double]]] zeroinitializer
@GlobI = local_unnamed_addr global [54 x [54 x [54 x double]]] zeroinitializer
@GlobJ = local_unnamed_addr global [54 x [54 x [54 x double]]] zeroinitializer
@GlobL = local_unnamed_addr global [1000 x [1000 x double]] zeroinitializer

define void @test(ptr noalias readonly captures(none) %0, ptr noalias readonly captures(none) %1, ptr noalias captures(none) %2, ptr noalias captures(none) %3, ptr noalias readonly captures(none) %4, ptr noalias readonly captures(none) %5, ptr noalias readonly captures(none) %6, ptr noalias readonly captures(none) %7, ptr noalias readonly captures(none) %8, ptr noalias readonly captures(none) %9) {
entry:
  %17 = load i32, ptr %7, align 4
  %18 = sext i32 %17 to i64
  %20 = load i32, ptr %8, align 4
  %21 = sext i32 %20 to i64
  %cmp1 = icmp sgt i32 %17, 0
  %cmp2 = icmp sgt i32 %20, 0
  %cond = and i1 %cmp1, %cmp2
  br i1 %cond, label %preheader, label %exit

preheader:
  br label %i.header

i.header:
  %i = phi i64 [ %i.next, %i.latch ], [ 1, %preheader ]
  %92 = add nsw i64 -55, %i
  %93 = add nsw i64 %i, -1
  %94 = mul nsw i64 %93, %21
  %invariant.gep = getelementptr double, ptr @GlobL, i64 %94
  br label %j.header

j.header:
  %j = phi i64 [ %j.next, %j.latch ], [ 1, %i.header ]
  %95 = add nsw i64 -55, %j
  %gep358 = getelementptr double, ptr %invariant.gep, i64 %j
  br label %IL.header

IL.header:
  %IL = phi i64 [ %IL.next, %IL.latch ], [ 1, %j.header ]
  %96 = mul nuw nsw i64 %IL, 54
  %97 = add nsw i64 %92, %96
  %98 = getelementptr double, ptr @GlobC, i64 %97
  %99 = load double, ptr %98, align 8
  %100 = getelementptr double, ptr @GlobG, i64 %97
  %101 = load double, ptr %100, align 8
  %102 = getelementptr double, ptr @GlobE, i64 %97
  %103 = load double, ptr %102, align 8
  %104 = getelementptr double, ptr @GlobI, i64 %97
  %105 = load double, ptr %104, align 8
  %106 = add nsw i64 %IL, -1
  %107 = mul nsw i64 %106, %21
  br label %JL.body

JL.body:
  %JL = phi i64 [ %JL.next, %JL.body ], [ 1, %IL.header ]
  %109 = mul nuw nsw i64 %JL, 54
  %110 = add nsw i64 %95, %109
  %111 = getelementptr double, ptr @GlobD, i64 %110
  %112 = load double, ptr %111, align 8
  %113 = fmul fast double %112, %99
  %114 = getelementptr double, ptr @GlobH, i64 %110
  %115 = load double, ptr %114, align 8
  %116 = fmul fast double %115, %101
  %117 = fadd fast double %116, %113
  %118 = getelementptr double, ptr @GlobF, i64 %110
  %119 = load double, ptr %118, align 8
  %120 = fmul fast double %119, %103
  %121 = fadd fast double %117, %120
  %122 = getelementptr double, ptr @GlobJ, i64 %110
  %123 = load double, ptr %122, align 8
  %124 = fmul fast double %123, %105
  %125 = fadd fast double %121, %124
  %126 = add nsw i64 %JL, %107
  %.idx247.us.us.us.us.us.us = mul nsw i64 %126, 8000
  %gep.us.us.us.us.us.us = getelementptr i8, ptr %gep358, i64 %.idx247.us.us.us.us.us.us
  %127 = getelementptr i8, ptr %gep.us.us.us.us.us.us, i64 -8008
  store double %125, ptr %127, align 8
  %JL.next = add nuw nsw i64 %JL, 1
  %exitcond.not = icmp eq i64 %JL, %21
  br i1 %exitcond.not, label %IL.latch, label %JL.body

IL.latch:
  %IL.next = add nuw nsw i64 %IL, 1
  %exitcond320.not = icmp eq i64 %IL, %18
  br i1 %exitcond320.not, label %j.latch, label %IL.header

j.latch:
  %j.next = add nuw nsw i64 %j, 1
  %exitcond324.not = icmp eq i64 %j, %21
  br i1 %exitcond324.not, label %i.latch, label %j.header

i.latch:
  %i.next = add nuw nsw i64 %i, 1
  %exitcond328.not = icmp eq i64 %i, %18
  br i1 %exitcond328.not, label %exit, label %i.header

exit:
  ret void
}


