; RUN: opt -S -loop-unroll -unroll-runtime %s -o - | FileCheck %s
; RUN: opt -S -loop-unroll -unroll-runtime -unroll-max-upperbound=6 %s -o - | FileCheck %s --check-prefix=UPPER

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

@global = dso_local local_unnamed_addr global i32 0, align 4
@global.1 = dso_local local_unnamed_addr global i8* null, align 4

; Check that loop in hoge_3, with a runtime upperbound of 3, is not unrolled.
; CHECK-LABEL: hoge_3
; CHECK: loop:
; CHECK: store
; CHECK-NOT: store
; CHECK: br i1 %{{.*}}, label %loop
; UPPER-LABEL: hoge_3
; UPPER: loop:
; UPPER: store
; UPPER-NOT: store
; UPPER: br i1 %{{.*}}, label %loop
define dso_local void @hoge_3(i8 %arg) {
entry:
  %x = load i32, i32* @global, align 4
  %y = load i8*, i8** @global.1, align 4
  %0 = icmp ult i32 %x, 17
  br i1 %0, label %loop, label %exit

loop:
  %iv = phi i32 [ %x, %entry ], [ %iv.next, %loop ]
  %ptr = phi i8* [ %y, %entry ], [ %ptr.next, %loop ]
  %iv.next = add nuw i32 %iv, 8
  %ptr.next = getelementptr inbounds i8, i8* %ptr, i32 1
  store i8 %arg, i8* %ptr.next, align 1
  %1 = icmp ult i32 %iv.next, 17
  br i1 %1, label %loop, label %exit

exit:
  ret void
}

; Check that loop in hoge_5, with a runtime upperbound of 5, is unrolled when -unroll-max-upperbound=4
; CHECK-LABEL: hoge_5
; CHECK: loop:
; CHECK: store
; CHECK-NOT: store
; CHECK: br i1 %{{.*}}, label %loop
; UPPER-LABEL: hoge_5
; UPPER: loop:
; UPPER: store
; UPPER: store
; UPPER: store
; UPPER: br i1 %{{.*}}, label %loop
define dso_local void @hoge_5(i8 %arg) {
entry:
  %x = load i32, i32* @global, align 4
  %y = load i8*, i8** @global.1, align 4
  %0 = icmp ult i32 %x, 17
  br i1 %0, label %loop, label %exit

loop:
  %iv = phi i32 [ %x, %entry ], [ %iv.next, %loop ]
  %ptr = phi i8* [ %y, %entry ], [ %ptr.next, %loop ]
  %iv.next = add nuw i32 %iv, 4
  %ptr.next = getelementptr inbounds i8, i8* %ptr, i32 1
  store i8 %arg, i8* %ptr.next, align 1
  %1 = icmp ult i32 %iv.next, 17
  br i1 %1, label %loop, label %exit

exit:
  ret void
}
