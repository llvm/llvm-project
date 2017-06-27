; RUN: opt < %s -basicaa -gvn -enable-load-pre -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define i32 @test1(i32* %p, i1 %C) {
; CHECK-LABEL: @test1(
block1:
	br i1 %C, label %block2, label %block3

block2:
 br label %block4
; CHECK: block2:
; CHECK-NEXT: load i32, i32* %p

block3:
  store i32 0, i32* %p
  br label %block4

block4:
  %PRE = load i32, i32* %p
  ret i32 %PRE
; CHECK: block4:
; CHECK-NEXT: phi i32
; CHECK-NEXT: ret i32
}

; This is a simple phi translation case.
define i32 @test2(i32* %p, i32* %q, i1 %C) {
; CHECK-LABEL: @test2(
block1:
	br i1 %C, label %block2, label %block3

block2:
 br label %block4
; CHECK: block2:
; CHECK-NEXT: load i32, i32* %q

block3:
  store i32 0, i32* %p
  br label %block4

block4:
  %P2 = phi i32* [%p, %block3], [%q, %block2]
  %PRE = load i32, i32* %P2
  ret i32 %PRE
; CHECK: block4:
; CHECK-NEXT: phi i32 [
; CHECK-NOT: load
; CHECK: ret i32
}

; This is a PRE case that requires phi translation through a GEP.
define i32 @test3(i32* %p, i32* %q, i32** %Hack, i1 %C) {
; CHECK-LABEL: @test3(
block1:
  %B = getelementptr i32, i32* %q, i32 1
  store i32* %B, i32** %Hack
	br i1 %C, label %block2, label %block3

block2:
 br label %block4
; CHECK: block2:
; CHECK-NEXT: load i32, i32* %B

block3:
  %A = getelementptr i32, i32* %p, i32 1
  store i32 0, i32* %A
  br label %block4

block4:
  %P2 = phi i32* [%p, %block3], [%q, %block2]
  %P3 = getelementptr i32, i32* %P2, i32 1
  %PRE = load i32, i32* %P3
  ret i32 %PRE
; CHECK: block4:
; CHECK: phi i32 [
; CHECK-NOT: load
; CHECK: ret i32
}

;; Here the loaded address is available, but the computation is in 'block3'
;; which does not dominate 'block2'.
define i32 @test4(i32* %p, i32* %q, i32** %Hack, i1 %C) {
; CHECK-LABEL: @test4(
block1:
	br i1 %C, label %block2, label %block3

block2:
 br label %block4
; CHECK: block2:
; CHECK:   load i32, i32*
; CHECK:   br label %block4

block3:
  %B = getelementptr i32, i32* %q, i32 1
  store i32* %B, i32** %Hack

  %A = getelementptr i32, i32* %p, i32 1
  store i32 0, i32* %A
  br label %block4

block4:
  %P2 = phi i32* [%p, %block3], [%q, %block2]
  %P3 = getelementptr i32, i32* %P2, i32 1
  %PRE = load i32, i32* %P3
  ret i32 %PRE
; CHECK: block4:
; CHECK: phi i32 [
; CHECK-NOT: load
; CHECK: ret i32
}

;void test5(int N, double *G) {
;  int j;
;  for (j = 0; j < N - 1; j++)
;    G[j] = G[j] + G[j+1];
;}

define void @test5(i32 %N, double* nocapture %G) nounwind ssp {
; CHECK-LABEL: @test5(
entry:
  %0 = add i32 %N, -1           
  %1 = icmp sgt i32 %0, 0       
  br i1 %1, label %bb.nph, label %return

bb.nph:                         
  %tmp = zext i32 %0 to i64     
  br label %bb

; CHECK: bb.nph:
; CHECK: load double, double*
; CHECK: br label %bb

bb:             
  %indvar = phi i64 [ 0, %bb.nph ], [ %tmp6, %bb ]
  %tmp6 = add i64 %indvar, 1                    
  %scevgep = getelementptr double, double* %G, i64 %tmp6
  %scevgep7 = getelementptr double, double* %G, i64 %indvar
  %2 = load double, double* %scevgep7, align 8
  %3 = load double, double* %scevgep, align 8 
  %4 = fadd double %2, %3             
  store double %4, double* %scevgep7, align 8
  %exitcond = icmp eq i64 %tmp6, %tmp 
  br i1 %exitcond, label %return, label %bb

; Should only be one load in the loop.
; CHECK: bb:
; CHECK: load double, double*
; CHECK-NOT: load double, double*
; CHECK: br i1 %exitcond

return:                               
  ret void
}

;void test6(int N, double *G) {
;  int j;
;  for (j = 0; j < N - 1; j++)
;    G[j+1] = G[j] + G[j+1];
;}

define void @test6(i32 %N, double* nocapture %G) nounwind ssp {
; CHECK-LABEL: @test6(
entry:
  %0 = add i32 %N, -1           
  %1 = icmp sgt i32 %0, 0       
  br i1 %1, label %bb.nph, label %return

bb.nph:                         
  %tmp = zext i32 %0 to i64     
  br label %bb

; CHECK: bb.nph:
; CHECK: load double, double*
; CHECK: br label %bb

bb:             
  %indvar = phi i64 [ 0, %bb.nph ], [ %tmp6, %bb ]
  %tmp6 = add i64 %indvar, 1                    
  %scevgep = getelementptr double, double* %G, i64 %tmp6
  %scevgep7 = getelementptr double, double* %G, i64 %indvar
  %2 = load double, double* %scevgep7, align 8
  %3 = load double, double* %scevgep, align 8 
  %4 = fadd double %2, %3             
  store double %4, double* %scevgep, align 8
  %exitcond = icmp eq i64 %tmp6, %tmp 
  br i1 %exitcond, label %return, label %bb

; Should only be one load in the loop.
; CHECK: bb:
; CHECK: load double, double*
; CHECK-NOT: load double, double*
; CHECK: br i1 %exitcond

return:                               
  ret void
}

;void test7(int N, double* G) {
;  long j;
;  G[1] = 1;
;  for (j = 1; j < N - 1; j++)
;      G[j+1] = G[j] + G[j+1];
;}

; This requires phi translation of the adds.
define void @test7(i32 %N, double* nocapture %G) nounwind ssp {
entry:
  %0 = getelementptr inbounds double, double* %G, i64 1   
  store double 1.000000e+00, double* %0, align 8
  %1 = add i32 %N, -1                             
  %2 = icmp sgt i32 %1, 1                         
  br i1 %2, label %bb.nph, label %return

bb.nph:                                           
  %tmp = sext i32 %1 to i64                       
  %tmp7 = add i64 %tmp, -1                        
  br label %bb

bb:                                               
  %indvar = phi i64 [ 0, %bb.nph ], [ %tmp9, %bb ] 
  %tmp8 = add i64 %indvar, 2                      
  %scevgep = getelementptr double, double* %G, i64 %tmp8  
  %tmp9 = add i64 %indvar, 1                      
  %scevgep10 = getelementptr double, double* %G, i64 %tmp9 
  %3 = load double, double* %scevgep10, align 8           
  %4 = load double, double* %scevgep, align 8             
  %5 = fadd double %3, %4                         
  store double %5, double* %scevgep, align 8
  %exitcond = icmp eq i64 %tmp9, %tmp7            
  br i1 %exitcond, label %return, label %bb

; Should only be one load in the loop.
; CHECK: bb:
; CHECK: load double, double*
; CHECK-NOT: load double, double*
; CHECK: br i1 %exitcond

return:                                           
  ret void
}

;; Here the loaded address isn't available in 'block2' at all, requiring a new
;; GEP to be inserted into it.
define i32 @test8(i32* %p, i32* %q, i32** %Hack, i1 %C) {
; CHECK-LABEL: @test8(
block1:
	br i1 %C, label %block2, label %block3

block2:
 br label %block4
; CHECK: block2:
; CHECK:   load i32, i32*
; CHECK:   br label %block4

block3:
  %A = getelementptr i32, i32* %p, i32 1
  store i32 0, i32* %A
  br label %block4

block4:
  %P2 = phi i32* [%p, %block3], [%q, %block2]
  %P3 = getelementptr i32, i32* %P2, i32 1
  %PRE = load i32, i32* %P3
  ret i32 %PRE
; CHECK: block4:
; CHECK: phi i32 [
; CHECK-NOT: load
; CHECK: ret i32
}

;void test9(int N, double* G) {
;  long j;
;  for (j = 1; j < N - 1; j++)
;      G[j+1] = G[j] + G[j+1];
;}

; This requires phi translation of the adds.
define void @test9(i32 %N, double* nocapture %G) nounwind ssp {
entry:
  add i32 0, 0
  %1 = add i32 %N, -1                             
  %2 = icmp sgt i32 %1, 1                         
  br i1 %2, label %bb.nph, label %return

bb.nph:                                           
  %tmp = sext i32 %1 to i64                       
  %tmp7 = add i64 %tmp, -1                        
  br label %bb

; CHECK: bb.nph:
; CHECK:   load double, double*
; CHECK:   br label %bb

bb:                                               
  %indvar = phi i64 [ 0, %bb.nph ], [ %tmp9, %bb ] 
  %tmp8 = add i64 %indvar, 2                      
  %scevgep = getelementptr double, double* %G, i64 %tmp8  
  %tmp9 = add i64 %indvar, 1                      
  %scevgep10 = getelementptr double, double* %G, i64 %tmp9 
  %3 = load double, double* %scevgep10, align 8           
  %4 = load double, double* %scevgep, align 8             
  %5 = fadd double %3, %4                         
  store double %5, double* %scevgep, align 8
  %exitcond = icmp eq i64 %tmp9, %tmp7            
  br i1 %exitcond, label %return, label %bb

; Should only be one load in the loop.
; CHECK: bb:
; CHECK: load double, double*
; CHECK-NOT: load double, double*
; CHECK: br i1 %exitcond

return:                                           
  ret void
}

;void test10(int N, double* G) {
;  long j;
;  for (j = 1; j < N - 1; j++)
;      G[j] = G[j] + G[j+1] + G[j-1];
;}

; PR5501
define void @test10(i32 %N, double* nocapture %G) nounwind ssp {
entry:
  %0 = add i32 %N, -1
  %1 = icmp sgt i32 %0, 1
  br i1 %1, label %bb.nph, label %return

bb.nph:
  %tmp = sext i32 %0 to i64
  %tmp8 = add i64 %tmp, -1
  br label %bb
; CHECK: bb.nph:
; CHECK:   load double, double*
; CHECK:   load double, double*
; CHECK:   br label %bb


bb:
  %indvar = phi i64 [ 0, %bb.nph ], [ %tmp11, %bb ]
  %scevgep = getelementptr double, double* %G, i64 %indvar
  %tmp9 = add i64 %indvar, 2
  %scevgep10 = getelementptr double, double* %G, i64 %tmp9
  %tmp11 = add i64 %indvar, 1
  %scevgep12 = getelementptr double, double* %G, i64 %tmp11
  %2 = load double, double* %scevgep12, align 8
  %3 = load double, double* %scevgep10, align 8
  %4 = fadd double %2, %3
  %5 = load double, double* %scevgep, align 8
  %6 = fadd double %4, %5
  store double %6, double* %scevgep12, align 8
  %exitcond = icmp eq i64 %tmp11, %tmp8
  br i1 %exitcond, label %return, label %bb

; Should only be one load in the loop.
; CHECK: bb:
; CHECK: load double, double*
; CHECK-NOT: load double, double*
; CHECK: br i1 %exitcond

return:
  ret void
}

; Test critical edge splitting.
define i32 @test11(i32* %p, i1 %C, i32 %N) {
; CHECK-LABEL: @test11(
block1:
        br i1 %C, label %block2, label %block3

block2:
 %cond = icmp sgt i32 %N, 1
 br i1 %cond, label %block4, label %block5
; CHECK: load i32, i32* %p
; CHECK-NEXT: br label %block4

block3:
  store i32 0, i32* %p
  br label %block4

block4:
  %PRE = load i32, i32* %p
  br label %block5

block5:
  %ret = phi i32 [ 0, %block2 ], [ %PRE, %block4 ]
  ret i32 %ret
; CHECK: block4:
; CHECK-NEXT: phi i32
}

declare void @f()
declare void @g(i32)
declare i32 @__CxxFrameHandler3(...)

; Test that loads aren't PRE'd into EH pads.
define void @test12(i32* %p) personality i32 (...)* @__CxxFrameHandler3 {
; CHECK-LABEL: @test12(
block1:
  invoke void @f()
          to label %block2 unwind label %catch.dispatch

block2:
  invoke void @f()
          to label %block3 unwind label %cleanup

block3:
  ret void

catch.dispatch:
  %cs1 = catchswitch within none [label %catch] unwind label %cleanup2

catch:
  %c = catchpad within %cs1 []
  catchret from %c to label %block2

cleanup:
  %c1 = cleanuppad within none []
  store i32 0, i32* %p
  cleanupret from %c1 unwind label %cleanup2

; CHECK: cleanup2:
; CHECK-NOT: phi
; CHECK-NEXT: %c2 = cleanuppad within none []
; CHECK-NEXT: %NOTPRE = load i32, i32* %p
cleanup2:
  %c2 = cleanuppad within none []
  %NOTPRE = load i32, i32* %p
  call void @g(i32 %NOTPRE)
  cleanupret from %c2 unwind to caller
}
