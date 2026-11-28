; RUN: llc -march=hexagon -disable-block-placement=0 -prevent-duplex-sep-v2=false\
; RUN: -hexagon-bit=0 -hexagon-initial-cfg-cleanup=0 -hexagon-instsimplify=0 \
; RUN: -hoist-const-loads=false < %s | FileCheck %s
; REQUIRES: asserts
; Check that compound is formed.
; CHECK: p{{[0-9]}} = cmp.eq(r{{[0-9]*}},#-1); if (p{{[0-9]}}.new) jump{{.*}} .LBB

target triple = "hexagon"
%s.4 = type { %s.5 }
%s.5 = type { i32, ptr }
%s.45 = type { i32, [0 x %s.46] }
%s.46 = type { i32, i32, i32, i32, i32, i32 }

@g0 = external constant [14 x %s.4], section ".rodata.msg", align 16

; Function Attrs: nounwind
declare void @f0(ptr, i32) #0

; Function Attrs: nounwind ssp
define zeroext i8 @f1(ptr nocapture readnone %a0, ptr readonly %a1, ptr readonly %a2, ptr %a3, i32 %a4) #1 align 2 {
b0:
  br i1 false, label %b16, label %b1

b1:                                               ; preds = %b0
  br i1 poison, label %b16, label %b2

b2:                                               ; preds = %b1
  br i1 poison, label %b16, label %b3

b3:                                               ; preds = %b2
  br i1 poison, label %b16, label %b4

b4:                                               ; preds = %b3
  br i1 poison, label %b6, label %b5

b5:                                               ; preds = %b4
  unreachable

b6:                                               ; preds = %b4
  br i1 poison, label %b8, label %b7

b7:                                               ; preds = %b6
  unreachable

b8:                                               ; preds = %b6
  br i1 poison, label %b9, label %b24

b9:                                               ; preds = %b8
  %v0 = load ptr, ptr poison, align 4
  %v1 = icmp ne ptr %v0, null
  %v2 = and i1 %v1, poison
  br i1 %v2, label %b10, label %b23

b10:                                              ; preds = %b9
  br label %b11

b11:                                              ; preds = %b22, %b10
  %v3 = phi i8 [ 0, %b10 ], [ %v24, %b22 ]
  %v4 = load i32, ptr null, align 4
  %v5 = icmp eq i32 %v4, %a4
  %v6 = or i1 %v5, false
  br i1 %v6, label %b13, label %b12

b12:                                              ; preds = %b11
  unreachable

b13:                                              ; preds = %b11
  %v7 = icmp eq i32 %v4, -1
  br i1 %v7, label %b16, label %b14

b14:                                              ; preds = %b13
  %v8 = zext i8 %v3 to i32
  %v9 = load ptr, ptr poison, align 4
  %v10 = getelementptr inbounds %s.45, ptr %v9, i32 0, i32 1, i32 %v8, i32 4
  %v11 = load i32, ptr %v10, align 4
  %v12 = load i8, ptr poison, align 1
  %v13 = zext i8 %v12 to i32
  %v14 = icmp eq i32 %v11, %v13
  br i1 %v14, label %b21, label %b15

b15:                                              ; preds = %b14
  br label %b18

b16:                                              ; preds = %b21, %b13, %b3, %b2, %b1, %b0
  %v15 = phi ptr [ @g0, %b0 ], [ getelementptr inbounds ([14 x %s.4], ptr @g0, i32 0, i32 1), %b1 ], [ getelementptr inbounds ([14 x %s.4], ptr @g0, i32 0, i32 2), %b2 ], [ getelementptr inbounds ([14 x %s.4], ptr @g0, i32 0, i32 3), %b3 ], [ getelementptr inbounds ([14 x %s.4], ptr @g0, i32 0, i32 8), %b13 ], [ getelementptr inbounds ([14 x %s.4], ptr @g0, i32 0, i32 10), %b21 ]
  call void @f0(ptr %v15, i32 poison) #0
  br label %b26

b17:                                              ; preds = %b19
  br i1 false, label %b21, label %b18

b18:                                              ; preds = %b17, %b15
  %v20 = phi i8 [ %v3, %b15 ], [ %v21, %b17 ]
  br i1 false, label %b21, label %b19

b19:                                              ; preds = %b18
  %v21 = add i8 %v20, 1
  br i1 false, label %b20, label %b17

b20:                                              ; preds = %b19
  br label %b21

b21:                                              ; preds = %b20, %b18, %b17, %b14
  %v22 = load i32, ptr poison, align 4
  %v23 = icmp eq i32 %v22, -1
  br i1 %v23, label %b16, label %b22

b22:                                              ; preds = %b21
  %v24 = add i8 0, 1
  br label %b11

b23:                                              ; preds = %b9
  br label %b26

b24:                                              ; preds = %b8
  br i1 poison, label %b25, label %b26

b25:                                              ; preds = %b24
  br label %b26

b26:                                              ; preds = %b25, %b24, %b23, %b16
  ret i8 poison
}

attributes #0 = { nounwind }
attributes #1 = { nounwind ssp }
