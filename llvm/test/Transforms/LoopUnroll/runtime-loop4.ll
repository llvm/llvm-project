; RUN: opt < %s -S -O2 -unroll-runtime=true -unroll-runtime-epilog=true  | FileCheck %s -check-prefix=EPILOG
; RUN: opt < %s -S -O2 -unroll-runtime=true -unroll-runtime-epilog=false | FileCheck %s -check-prefix=PROLOG

; Check runtime unrolling prologue can be promoted by LICM pass.

; EPILOG: entry:
; EPILOG: %xtraiter
; EPILOG: %lcmp.mod
; EPILOG: loop1:
; EPILOG: br i1 %lcmp.mod
; EPILOG: loop2.epil:

; PROLOG: entry:
; PROLOG: %xtraiter
; PROLOG: %lcmp.mod
; PROLOG: loop1:
; PROLOG: br i1 %lcmp.mod
; PROLOG: loop2.prol:

define void @unroll(i32 %iter, ptr %addr1, ptr %addr2) nounwind {
entry:
  br label %loop1

loop1:
  %iv1 = phi i32 [ 0, %entry ], [ %inc1, %loop1.latch ]
  %offset1 = getelementptr i32, ptr %addr1, i32 %iv1
  store i32 %iv1, ptr %offset1, align 4
  br label %loop2.header

loop2.header:
  %e = icmp uge i32 %iter, 1
  br i1 %e, label %loop2, label %exit2

loop2:
  %iv2 = phi i32 [ 0, %loop2.header ], [ %inc2, %loop2 ]
  %offset2 = getelementptr i32, ptr %addr2, i32 %iv2
  store i32 %iv2, ptr %offset2, align 4
  %inc2 = add i32 %iv2, 1
  %exitcnd2 = icmp uge i32 %inc2, %iter
  br i1 %exitcnd2, label %exit2, label %loop2

exit2:
  br label %loop1.latch

loop1.latch:
  %inc1 = add i32 %iv1, 1
  %exitcnd1 = icmp uge i32 %inc1, 1024
  br i1 %exitcnd1, label %exit, label %loop1

exit:
  ret void
}
