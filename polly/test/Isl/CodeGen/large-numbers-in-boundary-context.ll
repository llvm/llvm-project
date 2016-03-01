; RUN: opt %loadPolly -S -polly-codegen < %s | FileCheck %s
;
; The boundary context contains a constant that does not fit in 64 bits. Hence,
; we will check that we use an appropriaty typed constant, here with 65 bits.
; An alternative would be to bail out early but that would not be as easy.
;
; CHECK: {{.*}} = icmp sle i65 {{.*}}, -9223372036854775810
;
; CHECK: polly.start
;
target triple = "x86_64-unknown-linux-gnu"

@global = external global i32, align 4
@global1 = external global i32, align 4

; Function Attrs: nounwind uwtable
define void @hoge(i8* %arg) #0 {
bb:
  br label %bb5

bb5:                                              ; preds = %bb
  %tmp = load i32, i32* @global, align 4
  %tmp6 = sext i32 %tmp to i64
  br label %bb11

bb7:                                              ; preds = %bb19
  %tmp8 = load i32, i32* @global1, align 4
  %tmp9 = sext i32 %tmp8 to i64
  %tmp10 = icmp slt i64 %tmp13, %tmp9
  br i1 %tmp10, label %bb11, label %bb20

bb11:                                             ; preds = %bb7, %bb5
  %tmp12 = phi i64 [ %tmp6, %bb5 ], [ %tmp13, %bb7 ]
  %tmp13 = add i64 %tmp12, 1
  %tmp14 = getelementptr inbounds i8, i8* %arg, i64 %tmp13
  %tmp15 = load i8, i8* %tmp14, align 1
  br i1 false, label %bb16, label %bb17

bb16:                                             ; preds = %bb11
  br label %bb18

bb17:                                             ; preds = %bb11
  br label %bb18

bb18:                                             ; preds = %bb17, %bb16
  br label %bb19

bb19:                                             ; preds = %bb19, %bb18
  br i1 undef, label %bb19, label %bb7

bb20:                                             ; preds = %bb7
  br label %bb21

bb21:                                             ; preds = %bb20
  ret void
}
