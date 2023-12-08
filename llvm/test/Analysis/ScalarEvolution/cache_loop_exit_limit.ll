; RUN: opt -scalar-evolution-max-arith-depth=4 -scalar-evolution-max-add-rec-size=4 -loop-reduce -S < %s | FileCheck %s

; Check that the test does not hang.
define void @test_01(ptr nocapture %a) local_unnamed_addr {

; CHECK-LABEL: @test_01(

while.body.outer:
  %arrayidx2 = getelementptr inbounds i32, ptr %a, i64 96
  %arrayidx2.promoted50 = load i32, ptr %arrayidx2, align 1
  %a.promoted = load i32, ptr %a, align 1
  %add347.peel = xor i32 %arrayidx2.promoted50, -1
  %tobool48.peel = icmp eq i32 %a.promoted, %add347.peel
  br i1 %tobool48.peel, label %while.body.preheader, label %while.body4.preheader

while.body.preheader:                             ; preds = %while.body.outer
  %tobool48 = icmp eq i32 %a.promoted, 1
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  br i1 %tobool48, label %while.body, label %while.body4.preheader.loopexit

while.body4.preheader.loopexit:                   ; preds = %while.body
  br label %while.body4.preheader

while.body4.preheader:                            ; preds = %while.body4.preheader.loopexit, %while.body.outer
  br label %while.body4

while.body4:                                      ; preds = %while.body4.preheader, %while.end.22
  %0 = phi i32 [ %mul.22, %while.end.22 ], [ %arrayidx2.promoted50, %while.body4.preheader ]
  %mul = mul nsw i32 %0, %0
  br label %while.cond17

while.cond17:                                     ; preds = %while.cond17, %while.body4
  %add22.sink = phi i32 [ %add22, %while.cond17 ], [ %mul, %while.body4 ]
  %cmp = icmp slt i32 %add22.sink, 0
  %add22 = add nsw i32 %add22.sink, 1024
  br i1 %cmp, label %while.cond17, label %while.end

while.end:                                        ; preds = %while.cond17
  %mul.1 = mul nsw i32 %add22.sink, %add22.sink
  br label %while.cond17.1

while.cond17.1:                                   ; preds = %while.cond17.1, %while.end
  %add22.sink.1 = phi i32 [ %add22.1, %while.cond17.1 ], [ %mul.1, %while.end ]
  %cmp.1 = icmp slt i32 %add22.sink.1, 0
  %add22.1 = add nsw i32 %add22.sink.1, 2048
  br i1 %cmp.1, label %while.cond17.1, label %while.end.1

while.end.1:                                      ; preds = %while.cond17.1
  %mul.2 = mul nsw i32 %add22.sink.1, %add22.sink.1
  br label %while.cond17.2

while.cond17.2:                                   ; preds = %while.cond17.2, %while.end.1
  %add22.sink.2 = phi i32 [ %add22.2, %while.cond17.2 ], [ %mul.2, %while.end.1 ]
  %cmp.2 = icmp slt i32 %add22.sink.2, 0
  %add22.2 = add nsw i32 %add22.sink.2, 4096
  br i1 %cmp.2, label %while.cond17.2, label %while.end.2

while.end.2:                                      ; preds = %while.cond17.2
  %mul.3 = mul nsw i32 %add22.sink.2, %add22.sink.2
  br label %while.cond17.3

while.cond17.3:                                   ; preds = %while.cond17.3, %while.end.2
  %add22.sink.3 = phi i32 [ %add22.3, %while.cond17.3 ], [ %mul.3, %while.end.2 ]
  %cmp.3 = icmp slt i32 %add22.sink.3, 0
  %add22.3 = add nsw i32 %add22.sink.3, 8192
  br i1 %cmp.3, label %while.cond17.3, label %while.end.3

while.end.3:                                      ; preds = %while.cond17.3
  %mul.4 = mul nsw i32 %add22.sink.3, %add22.sink.3
  br label %while.cond17.4

while.cond17.4:                                   ; preds = %while.cond17.4, %while.end.3
  %add22.sink.4 = phi i32 [ %add22.4, %while.cond17.4 ], [ %mul.4, %while.end.3 ]
  %cmp.4 = icmp slt i32 %add22.sink.4, 0
  %add22.4 = add nsw i32 %add22.sink.4, 16384
  br i1 %cmp.4, label %while.cond17.4, label %while.end.4

while.end.4:                                      ; preds = %while.cond17.4
  %mul.5 = mul nsw i32 %add22.sink.4, %add22.sink.4
  br label %while.cond17.5

while.cond17.5:                                   ; preds = %while.cond17.5, %while.end.4
  %add22.sink.5 = phi i32 [ %add22.5, %while.cond17.5 ], [ %mul.5, %while.end.4 ]
  %cmp.5 = icmp slt i32 %add22.sink.5, 0
  %add22.5 = add nsw i32 %add22.sink.5, 32768
  br i1 %cmp.5, label %while.cond17.5, label %while.end.5

while.end.5:                                      ; preds = %while.cond17.5
  %mul.6 = mul nsw i32 %add22.sink.5, %add22.sink.5
  br label %while.cond17.6

while.cond17.6:                                   ; preds = %while.cond17.6, %while.end.5
  %add22.sink.6 = phi i32 [ %add22.6, %while.cond17.6 ], [ %mul.6, %while.end.5 ]
  %cmp.6 = icmp slt i32 %add22.sink.6, 0
  %add22.6 = add nsw i32 %add22.sink.6, 65536
  br i1 %cmp.6, label %while.cond17.6, label %while.end.6

while.end.6:                                      ; preds = %while.cond17.6
  %mul.7 = mul nsw i32 %add22.sink.6, %add22.sink.6
  br label %while.cond17.7

while.cond17.7:                                   ; preds = %while.cond17.7, %while.end.6
  %add22.sink.7 = phi i32 [ %add22.7, %while.cond17.7 ], [ %mul.7, %while.end.6 ]
  %cmp.7 = icmp slt i32 %add22.sink.7, 0
  %add22.7 = add nsw i32 %add22.sink.7, 131072
  br i1 %cmp.7, label %while.cond17.7, label %while.end.7

while.end.7:                                      ; preds = %while.cond17.7
  %mul.8 = mul nsw i32 %add22.sink.7, %add22.sink.7
  br label %while.cond17.8

while.cond17.8:                                   ; preds = %while.cond17.8, %while.end.7
  %add22.sink.8 = phi i32 [ %add22.8, %while.cond17.8 ], [ %mul.8, %while.end.7 ]
  %cmp.8 = icmp slt i32 %add22.sink.8, 0
  %add22.8 = add nsw i32 %add22.sink.8, 262144
  br i1 %cmp.8, label %while.cond17.8, label %while.end.8

while.end.8:                                      ; preds = %while.cond17.8
  %mul.9 = mul nsw i32 %add22.sink.8, %add22.sink.8
  br label %while.cond17.9

while.cond17.9:                                   ; preds = %while.cond17.9, %while.end.8
  %add22.sink.9 = phi i32 [ %add22.9, %while.cond17.9 ], [ %mul.9, %while.end.8 ]
  %cmp.9 = icmp slt i32 %add22.sink.9, 0
  %add22.9 = add nsw i32 %add22.sink.9, 524288
  br i1 %cmp.9, label %while.cond17.9, label %while.end.9

while.end.9:                                      ; preds = %while.cond17.9
  %mul.10 = mul nsw i32 %add22.sink.9, %add22.sink.9
  br label %while.cond17.10

while.cond17.10:                                  ; preds = %while.cond17.10, %while.end.9
  %add22.sink.10 = phi i32 [ %add22.10, %while.cond17.10 ], [ %mul.10, %while.end.9 ]
  %cmp.10 = icmp slt i32 %add22.sink.10, 0
  %add22.10 = add nsw i32 %add22.sink.10, 1048576
  br i1 %cmp.10, label %while.cond17.10, label %while.end.10

while.end.10:                                     ; preds = %while.cond17.10
  %mul.11 = mul nsw i32 %add22.sink.10, %add22.sink.10
  br label %while.cond17.11

while.cond17.11:                                  ; preds = %while.cond17.11, %while.end.10
  %add22.sink.11 = phi i32 [ %add22.11, %while.cond17.11 ], [ %mul.11, %while.end.10 ]
  %cmp.11 = icmp slt i32 %add22.sink.11, 0
  %add22.11 = add nsw i32 %add22.sink.11, 2097152
  br i1 %cmp.11, label %while.cond17.11, label %while.end.11

while.end.11:                                     ; preds = %while.cond17.11
  %mul.12 = mul nsw i32 %add22.sink.11, %add22.sink.11
  br label %while.cond17.12

while.cond17.12:                                  ; preds = %while.cond17.12, %while.end.11
  %add22.sink.12 = phi i32 [ %add22.12, %while.cond17.12 ], [ %mul.12, %while.end.11 ]
  %cmp.12 = icmp slt i32 %add22.sink.12, 0
  %add22.12 = add nsw i32 %add22.sink.12, 4194304
  br i1 %cmp.12, label %while.cond17.12, label %while.end.12

while.end.12:                                     ; preds = %while.cond17.12
  %mul.13 = mul nsw i32 %add22.sink.12, %add22.sink.12
  br label %while.cond17.13

while.cond17.13:                                  ; preds = %while.cond17.13, %while.end.12
  %add22.sink.13 = phi i32 [ %add22.13, %while.cond17.13 ], [ %mul.13, %while.end.12 ]
  %cmp.13 = icmp slt i32 %add22.sink.13, 0
  %add22.13 = add nsw i32 %add22.sink.13, 8388608
  br i1 %cmp.13, label %while.cond17.13, label %while.end.13

while.end.13:                                     ; preds = %while.cond17.13
  %mul.14 = mul nsw i32 %add22.sink.13, %add22.sink.13
  br label %while.cond17.14

while.cond17.14:                                  ; preds = %while.cond17.14, %while.end.13
  %add22.sink.14 = phi i32 [ %add22.14, %while.cond17.14 ], [ %mul.14, %while.end.13 ]
  %cmp.14 = icmp slt i32 %add22.sink.14, 0
  %add22.14 = add nsw i32 %add22.sink.14, 16777216
  br i1 %cmp.14, label %while.cond17.14, label %while.end.14

while.end.14:                                     ; preds = %while.cond17.14
  %mul.15 = mul nsw i32 %add22.sink.14, %add22.sink.14
  br label %while.cond17.15

while.cond17.15:                                  ; preds = %while.cond17.15, %while.end.14
  %add22.sink.15 = phi i32 [ %add22.15, %while.cond17.15 ], [ %mul.15, %while.end.14 ]
  %cmp.15 = icmp slt i32 %add22.sink.15, 0
  %add22.15 = add nsw i32 %add22.sink.15, 33554432
  br i1 %cmp.15, label %while.cond17.15, label %while.end.15

while.end.15:                                     ; preds = %while.cond17.15
  %mul.16 = mul nsw i32 %add22.sink.15, %add22.sink.15
  br label %while.cond17.16

while.cond17.16:                                  ; preds = %while.cond17.16, %while.end.15
  %add22.sink.16 = phi i32 [ %add22.16, %while.cond17.16 ], [ %mul.16, %while.end.15 ]
  %cmp.16 = icmp slt i32 %add22.sink.16, 0
  %add22.16 = add nsw i32 %add22.sink.16, 67108864
  br i1 %cmp.16, label %while.cond17.16, label %while.end.16

while.end.16:                                     ; preds = %while.cond17.16
  %mul.17 = mul nsw i32 %add22.sink.16, %add22.sink.16
  br label %while.cond17.17

while.cond17.17:                                  ; preds = %while.cond17.17, %while.end.16
  %add22.sink.17 = phi i32 [ %add22.17, %while.cond17.17 ], [ %mul.17, %while.end.16 ]
  %cmp.17 = icmp slt i32 %add22.sink.17, 0
  %add22.17 = add nsw i32 %add22.sink.17, 134217728
  br i1 %cmp.17, label %while.cond17.17, label %while.end.17

while.end.17:                                     ; preds = %while.cond17.17
  %mul.18 = mul nsw i32 %add22.sink.17, %add22.sink.17
  br label %while.cond17.18

while.cond17.18:                                  ; preds = %while.cond17.18, %while.end.17
  %add22.sink.18 = phi i32 [ %add22.18, %while.cond17.18 ], [ %mul.18, %while.end.17 ]
  %cmp.18 = icmp slt i32 %add22.sink.18, 0
  %add22.18 = add nsw i32 %add22.sink.18, 268435456
  br i1 %cmp.18, label %while.cond17.18, label %while.end.18

while.end.18:                                     ; preds = %while.cond17.18
  %mul.19 = mul nsw i32 %add22.sink.18, %add22.sink.18
  br label %while.cond17.19

while.cond17.19:                                  ; preds = %while.cond17.19, %while.end.18
  %add22.sink.19 = phi i32 [ %add22.19, %while.cond17.19 ], [ %mul.19, %while.end.18 ]
  %cmp.19 = icmp slt i32 %add22.sink.19, 0
  %add22.19 = add nsw i32 %add22.sink.19, 536870912
  br i1 %cmp.19, label %while.cond17.19, label %while.end.19

while.end.19:                                     ; preds = %while.cond17.19
  %mul.20 = mul nsw i32 %add22.sink.19, %add22.sink.19
  br label %while.cond17.20

while.cond17.20:                                  ; preds = %while.cond17.20, %while.end.19
  %add22.sink.20 = phi i32 [ %add22.20, %while.cond17.20 ], [ %mul.20, %while.end.19 ]
  %cmp.20 = icmp slt i32 %add22.sink.20, 0
  %add22.20 = add nsw i32 %add22.sink.20, 1073741824
  br i1 %cmp.20, label %while.cond17.20, label %while.end.20

while.end.20:                                     ; preds = %while.cond17.20
  %mul.21 = mul nsw i32 %add22.sink.20, %add22.sink.20
  br label %while.cond17.21

while.cond17.21:                                  ; preds = %while.cond17.21, %while.end.20
  %add22.sink.21 = phi i32 [ %add22.21, %while.cond17.21 ], [ %mul.21, %while.end.20 ]
  %cmp.21 = icmp slt i32 %add22.sink.21, 0
  %add22.21 = or i32 %add22.sink.21, -2147483648
  br i1 %cmp.21, label %while.cond17.21, label %while.end.22

while.end.22:                                     ; preds = %while.cond17.21
  %mul.22 = mul nsw i32 %add22.sink.21, %add22.sink.21
  br label %while.body4
}
