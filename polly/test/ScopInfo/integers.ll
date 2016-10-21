; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s

; Check that we correctly convert integers to isl values.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

; Large positive integer
define void @f(i1024* nocapture %a) nounwind {
entry:
  br label %bb

bb:
  %indvar = phi i1024 [ 0, %entry ], [ %indvar.next, %bb ]
  store i1024 %indvar, i1024* %a, align 8
  %indvar.next = add nsw i1024 %indvar, 1
  %exitcond = icmp eq i1024 %indvar, 123456000000000000000000000
; CHECK:  'bb => return' in function 'f'
; CHECK: i0 <= 123456000000000000000000000
  br i1 %exitcond, label %return, label %bb

return:
  ret void
}

; Normal positive integer
define void @f2(i32* nocapture %a) nounwind {
entry:
  br label %bb

bb:
  %indvar = phi i32 [ 0, %entry ], [ %indvar.next, %bb ]
  %scevgep = getelementptr i32, i32* %a, i32 %indvar
  store i32 %indvar, i32* %scevgep, align 8
  %indvar.next = add nsw i32 %indvar, 1
  %exitcond = icmp eq i32 %indvar, 123456
; CHECK:  'bb => return' in function 'f2'
; CHECK: i0 <= 123456
  br i1 %exitcond, label %return, label %bb

return:
  ret void
}

; Normal negative integer
define void @f3(i32* nocapture %a, i32 %n) nounwind {
entry:
  br label %bb

bb:
  %indvar = phi i32 [ 0, %entry ], [ %indvar.next, %bb ]
  %scevgep = getelementptr i32, i32* %a, i32 %indvar
  store i32 %indvar, i32* %scevgep, align 8
  %indvar.next = add nsw i32 %indvar, 1
  %sub = sub i32 %n, 123456
  %exitcond = icmp eq i32 %indvar, %sub
; CHECK:  'bb => return' in function 'f3'
; CHECK: -123456
  br i1 %exitcond, label %return, label %bb

return:
  ret void
}

; Large negative integer
define void @f4(i1024* nocapture %a, i1024 %n) nounwind {
entry:
  br label %bb

bb:
  %indvar = phi i1024 [ 0, %entry ], [ %indvar.next, %bb ]
  %scevgep = getelementptr i1024, i1024* %a, i1024 %indvar
  store i1024 %indvar, i1024* %scevgep, align 8
  %indvar.next = add nsw i1024 %indvar, 1
  %sub = sub i1024 %n, 123456000000000000000000000000000000
; CHECK:  'bb => return' in function 'f4'
; CHECK: -123456000000000000000000000000000000
  %exitcond = icmp eq i1024 %indvar, %sub
  br i1 %exitcond, label %return, label %bb

return:
  ret void
}

define void @f5(i1023* nocapture %a, i1023 %n) nounwind {
entry:
  br label %bb

bb:
  %indvar = phi i1023 [ 0, %entry ], [ %indvar.next, %bb ]
  %scevgep = getelementptr i1023, i1023* %a, i1023 %indvar
  store i1023 %indvar, i1023* %scevgep, align 8
  %indvar.next = add nsw i1023 %indvar, 1
  %sub = sub i1023 %n, 123456000000000000000000000000000000
; CHECK:  'bb => return' in function 'f5'
; CHECK: -123456000000000000000000000000000000
  %exitcond = icmp eq i1023 %indvar, %sub
  br i1 %exitcond, label %return, label %bb

return:
  ret void
}

; Tiny negative integer
define void @f6(i3* nocapture %a, i3 %n) nounwind {
entry:
  br label %bb

bb:
  %indvar = phi i3 [ 0, %entry ], [ %indvar.next, %bb ]
  %scevgep = getelementptr i3, i3* %a, i3 %indvar
  store i3 %indvar, i3* %scevgep, align 8
  %indvar.next = add nsw i3 %indvar, 1
  %sub = sub i3 %n, 3
; CHECK-LABEL:  'bb => return' in function 'f6'
; CHECK:         Context:
; CHECK-NEXT:    [n] -> {  : -4 <= n <= 3 }
; CHECK-NEXT:    Assumed Context:
; CHECK-NEXT:    [n] -> {  :  }
; CHECK-NEXT:    Invalid Context:
; CHECK-NEXT:    [n] -> {  : 1 = 0 }

; CHECK:     Statements {
; CHECK-NEXT:    Stmt_bb
; CHECK-NEXT:        Domain :=
; CHECK-NEXT:            [n] -> { Stmt_bb[i0] : i0 >= 0 and 8*floor((2 - n)/8) >= -5 - n + i0 and 8*floor((2 - n)/8) <= -2 - n };
; CHECK-NEXT:        Schedule :=
; CHECK-NEXT:            [n] -> { Stmt_bb[i0] -> [i0] };
; CHECK-NEXT:        MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:            [n] -> { Stmt_bb[i0] -> MemRef_a[i0] };
; CHECK-NEXT:}

  %exitcond = icmp eq i3 %indvar, %sub
  br i1 %exitcond, label %return, label %bb

return:
  ret void
}
