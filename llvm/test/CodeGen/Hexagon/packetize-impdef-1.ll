; RUN: llc -march=hexagon < %s | FileCheck %s
; REQUIRES: asserts

; Test that the compiler doesn't assert because IMPLICIT_DEF instructions are
; are added to the same packet as a use. This test case asserts if the
; IMPLICIT_DEFs are not handled properly.
;
; r0 = IMPLICIT_DEF
; r1 = IMPLICIT_DEF
; S2_storerd_io r29, 0, d0

; CHECK: memd(r29+#0) = r{{[0-9]+}}:{{[0-9]+}}
; CHECK: memd(r29+#0) = r{{[0-9]+}}:{{[0-9]+}}

define i8** @f0(i8* %a0) local_unnamed_addr {
b0:
  %v0 = tail call i8* @f1(i32 0)
  %v1 = tail call i8* @f1(i32 8)
  %v2 = bitcast i8* %v1 to i8**
  %v3 = load i32, i32* undef, align 4
  %v4 = tail call i8* @f4(i8* %a0, i32 0, i32 %v3)
  %v5 = sub nsw i32 %v3, 0
  br label %b1

b1:                                               ; preds = %b0
  switch i8 undef, label %b3 [
    i8 0, label %b4
    i8 92, label %b2
    i8 44, label %b4
  ]

b2:                                               ; preds = %b1
  unreachable

b3:                                               ; preds = %b1
  unreachable

b4:                                               ; preds = %b1, %b1
  br label %b5

b5:                                               ; preds = %b4
  br i1 undef, label %b27, label %b6

b6:                                               ; preds = %b5
  %v6 = ptrtoint i8* %v4 to i32
  %v7 = sub i32 0, %v6
  %v8 = call i8* @f4(i8* nonnull %v4, i32 0, i32 %v7)
  %v9 = call i8* @f4(i8* nonnull %v4, i32 undef, i32 %v5)
  br label %b7

b7:                                               ; preds = %b6
  br i1 undef, label %b8, label %b9

b8:                                               ; preds = %b7
  br label %b9

b9:                                               ; preds = %b8, %b7
  %v10 = phi i32 [ 2, %b8 ], [ 0, %b7 ]
  %v11 = load i8, i8* %v9, align 1
  switch i8 %v11, label %b12 [
    i8 43, label %b10
    i8 45, label %b10
  ]

b10:                                              ; preds = %b9, %b9
  br i1 undef, label %b11, label %b12

b11:                                              ; preds = %b10
  %v12 = call i64 @f6(i8* nonnull %v9, i8** nonnull undef, i32 10)
  %v13 = load i8*, i8** undef, align 4
  %v14 = ptrtoint i8* %v13 to i32
  br label %b15

b12:                                              ; preds = %b10, %b9
  switch i8 undef, label %b14 [
    i8 0, label %b13
    i8 46, label %b13
  ]

b13:                                              ; preds = %b12, %b12
  br label %b15

b14:                                              ; preds = %b12
  unreachable

b15:                                              ; preds = %b13, %b11
  %v15 = phi i32 [ undef, %b13 ], [ %v14, %b11 ]
  %v16 = phi i32 [ 2, %b13 ], [ 1, %b11 ]
  %v17 = phi i64 [ undef, %b13 ], [ %v12, %b11 ]
  %v18 = call i32* @f5()
  br label %b16

b16:                                              ; preds = %b15
  %v19 = icmp ne i32 %v10, %v16
  %v20 = or i1 undef, %v19
  br i1 %v20, label %b17, label %b18

b17:                                              ; preds = %b16
  call void @f2(i8* %v8)
  br label %b27

b18:                                              ; preds = %b16
  br i1 undef, label %b19, label %b20

b19:                                              ; preds = %b18
  br label %b24

b20:                                              ; preds = %b18
  %v21 = add i32 %v5, -2
  %v22 = sub i32 %v21, %v7
  %v23 = add i32 %v22, %v15
  %v24 = sub i32 %v23, 0
  br label %b21

b21:                                              ; preds = %b20
  %v25 = icmp ne i32 %v24, 2
  %v26 = and i1 %v25, undef
  br i1 %v26, label %b22, label %b23

b22:                                              ; preds = %b21
  unreachable

b23:                                              ; preds = %b21
  br label %b24

b24:                                              ; preds = %b23, %b19
  %v27 = phi i64 [ 0, %b19 ], [ %v17, %b23 ]
  br label %b25

b25:                                              ; preds = %b24
  %v28 = icmp sgt i64 undef, %v27
  br i1 %v28, label %b28, label %b26

b26:                                              ; preds = %b25
  unreachable

b27:                                              ; preds = %b17, %b5
  call void @f2(i8* %v4)
  call void @f2(i8* %v0)
  %v29 = call i8* @f3(i8* undef, i8* nonnull %a0)
  ret i8** %v2

b28:                                              ; preds = %b25
  call void @f2(i8* %v9)
  unreachable
}

declare i8* @f1(i32) local_unnamed_addr

declare void @f2(i8* nocapture) local_unnamed_addr

declare i8* @f3(i8*, i8* nocapture readonly) local_unnamed_addr

declare i8* @f4(i8*, i32, i32) local_unnamed_addr

declare i32* @f5() local_unnamed_addr

declare i64 @f6(i8*, i8**, i32) local_unnamed_addr
