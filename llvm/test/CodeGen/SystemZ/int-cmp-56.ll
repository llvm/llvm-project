; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s
;
; Check that signed comparisons against 0 are only eliminated if the "nsw"
; flag is present on the defining add (with register) instruction. For an
; equality comparison, add logical can be used.

define i32 @fun0(i32 %arg, i32 %arg2, i32 %arg3) {
; CHECK-LABEL: fun0:
; CHECK: jle .LBB0_2{{$}}
; CHECK: je .LBB0_4{{$}}

bb:
  %tmp2 = add nsw i32 %arg, %arg2
  %tmp3 = icmp sgt i32 %tmp2, 0
  br i1 %tmp3, label %bb3, label %bb1

bb1:
  %tmp4 = add nsw i32 %arg, %arg3
  %tmp5 = icmp eq i32 %tmp4, 0
  br i1 %tmp5, label %bb4, label %bb2

bb2:
  ret i32 0

bb3:
  ret i32 1

bb4:
  ret i32 2
}

; No "nsw" flag
define i32 @fun1(i32 %arg, i32 %arg2, i32 %arg3) {
; CHECK-LABEL: fun1:
; CHECK: cijle
; CHECK: alr
; CHECK: jhe

bb:
  %tmp2 = add i32 %arg, %arg2
  %tmp3 = icmp sgt i32 %tmp2, 0
  br i1 %tmp3, label %bb3, label %bb1

bb1:
  %tmp4 = add i32 %arg, %arg3
  %tmp5 = icmp eq i32 %tmp4, 0
  br i1 %tmp5, label %bb4, label %bb2

bb2:
  ret i32 0

bb3:
  ret i32 1

bb4:
  ret i32 2
}

; "nuw" flag
define i32 @fun2(i32 %arg, i32 %arg2, i32 %arg3) {
; CHECK-LABEL: fun2:
; CHECK: cijle
; CHECK: alr
; CHECK: jhe

bb:
  %tmp2 = add nuw i32 %arg, %arg2
  %tmp3 = icmp sgt i32 %tmp2, 0
  br i1 %tmp3, label %bb3, label %bb1

bb1:
  %tmp4 = add nuw i32 %arg, %arg3
  %tmp5 = icmp eq i32 %tmp4, 0
  br i1 %tmp5, label %bb4, label %bb2

bb2:
  ret i32 0

bb3:
  ret i32 1

bb4:
  ret i32 2
}

; Subtraction does not produce the value of zero in case of overflow, so
; "nsw" is not needed for the equality check against zero.
define i32 @fun3(i32 %arg, i32 %arg2, i32 %arg3) {
; CHECK-LABEL: fun3:
; CHECK: jle .LBB3_2{{$}}
; CHECK: je .LBB3_4{{$}}

bb:
  %tmp2 = sub nsw i32 %arg, %arg2
  %tmp3 = icmp sgt i32 %tmp2, 0
  br i1 %tmp3, label %bb3, label %bb1

bb1:
  %tmp4 = sub nsw i32 %arg, %arg3
  %tmp5 = icmp eq i32 %tmp4, 0
  br i1 %tmp5, label %bb4, label %bb2

bb2:
  ret i32 0

bb3:
  ret i32 1

bb4:
  ret i32 2
}

; No "nsw" flag
define i32 @fun4(i32 %arg, i32 %arg2, i32 %arg3) {
; CHECK-LABEL: fun4:
; CHECK: cijle
; CHECK: je .LBB4_4{{$}}

bb:
  %tmp2 = sub i32 %arg, %arg2
  %tmp3 = icmp sgt i32 %tmp2, 0
  br i1 %tmp3, label %bb3, label %bb1

bb1:
  %tmp4 = sub i32 %arg, %arg3
  %tmp5 = icmp eq i32 %tmp4, 0
  br i1 %tmp5, label %bb4, label %bb2

bb2:
  ret i32 0

bb3:
  ret i32 1

bb4:
  ret i32 2
}

; "nuw" flag
define i32 @fun5(i32 %arg, i32 %arg2, i32 %arg3) {
; CHECK-LABEL: fun5:
; CHECK: cijle
; CHECK: je .LBB5_4{{$}}

bb:
  %tmp2 = sub nuw i32 %arg, %arg2
  %tmp3 = icmp sgt i32 %tmp2, 0
  br i1 %tmp3, label %bb3, label %bb1

bb1:
  %tmp4 = sub nuw i32 %arg, %arg3
  %tmp5 = icmp eq i32 %tmp4, 0
  br i1 %tmp5, label %bb4, label %bb2

bb2:
  ret i32 0

bb3:
  ret i32 1

bb4:
  ret i32 2
}
