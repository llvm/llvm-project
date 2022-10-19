; RUN: llvm-reduce -abort-on-invalid-reduction --delta-passes=basic-blocks --test FileCheck --test-arg --check-prefix=CHECK-INTERESTINGNESS0 --test-arg %s --test-arg --input-file %s -o %t.0
; RUN: FileCheck -check-prefix=RESULT0 %s < %t.0

; RUN: llvm-reduce -abort-on-invalid-reduction --delta-passes=basic-blocks --test FileCheck --test-arg --check-prefix=CHECK-INTERESTINGNESS1 --test-arg %s --test-arg --input-file %s -o %t.1
; RUN: FileCheck -check-prefix=RESULT1 %s < %t.1

; CHECK-INTERESTINGNESS0: store i32 1,
; CHECK-INTERESTINGNESS0: store i32 2,

; CHECK-INTERESTINGNESS1: store i32 2,


; RESULT0: bb:
; RESULT0-NEXT: %bb.load = load i32, ptr null, align 4
; RESULT0-NEXT: store i32 0, ptr null, align 4
; RESULT0-NEXT: br i1 %arg0, label %bb1, label %bb2

; RESULT0: bb1:
; RESULT0-NEXT: %bb1.phi = phi i32 [ %bb.load, %bb ], [ %bb2.phi, %bb2 ], [ %bb2.phi, %bb2 ]
; RESULT0-NEXT: store i32 1, ptr null, align 4
; RESULT0-NEXT: ret void

; RESULT0: bb2: ; preds = %bb
; RESULT0-NEXT: %bb2.phi = phi i32 [ %bb.load, %bb ]
; RESULT0-NEXT: store i32 2, ptr null, align 4
; RESULT0-NEXT: switch i32 %bb2.phi, label %bb1 [
; RESULT0-NEXT: i32 0, label %bb1
; RESULT0-NEXT: ]


; RESULT1: bb:
; RESULT1-NEXT: %bb.load = load i32, ptr null, align 4
; RESULT1-NEXT: store i32 0, ptr null, align 4
; RESULT1-NEXT: br label %bb2

; RESULT1: bb2:
; RESULT1-NEXT: %bb2.phi = phi i32 [ %bb.load, %bb ]
; RESULT1-NEXT: store i32 2, ptr null, align 4
; RESULT1-NEXT: ret void
define void @main(i1 %arg0) {
bb:
  %bb.load = load i32, ptr null
  store i32 0, ptr null
  br i1 %arg0, label %bb1, label %bb2

bb1:
  %bb1.phi = phi i32 [%bb.load, %bb], [9, %bb3], [%bb2.phi, %bb2]
  store i32 1, ptr null
  ret void

bb2:
  %bb2.phi = phi i32 [%bb.load, %bb], [%bb3.load, %bb3]
  store i32 2, ptr null
  switch i32 %bb2.phi, label %bb3 [
    i32 0, label %bb1
    i32 1, label %bb4
  ]

bb3:
  %bb3.load = load i32, ptr null
  store i32 3, ptr null
  br i1 true, label %bb2, label %bb1

bb4:
  store i32 4, ptr null
  ret void
}
