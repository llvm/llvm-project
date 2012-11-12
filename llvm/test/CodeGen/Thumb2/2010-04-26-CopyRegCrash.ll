; RUN: llc < %s -mtriple=thumbv7-apple-darwin
; Radar 7896289

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

define void @test(i32 %mode) nounwind optsize noinline {
entry:
  br i1 undef, label %return, label %bb3

bb3:                                              ; preds = %entry
  br i1 undef, label %bb15, label %bb18

bb15:                                             ; preds = %bb3
  unreachable

bb18:                                             ; preds = %bb3
  switch i32 %mode, label %return [
    i32 0, label %bb26
    i32 1, label %bb56
    i32 2, label %bb107
    i32 6, label %bb150.preheader
    i32 9, label %bb310.preheader
    i32 13, label %bb414.preheader
    i32 15, label %bb468.preheader
    i32 16, label %bb522.preheader
  ]

bb150.preheader:                                  ; preds = %bb18
  br i1 undef, label %bb154, label %bb160

bb310.preheader:                                  ; preds = %bb18
  unreachable

bb414.preheader:                                  ; preds = %bb18
  unreachable

bb468.preheader:                                  ; preds = %bb18
  unreachable

bb522.preheader:                                  ; preds = %bb18
  unreachable

bb26:                                             ; preds = %bb18
  unreachable

bb56:                                             ; preds = %bb18
  unreachable

bb107:                                            ; preds = %bb18
  br label %bb110

bb110:                                            ; preds = %bb122, %bb107
  %asmtmp.i.i179 = tail call i16 asm "rev16 $0, $1\0A", "=l,l"(i16 undef) nounwind ; <i16> [#uses=1]
  %asmtmp.i.i178 = tail call i16 asm "rev16 $0, $1\0A", "=l,l"(i16 %asmtmp.i.i179) nounwind ; <i16> [#uses=1]
  store i16 %asmtmp.i.i178, i16* undef, align 2
  br i1 undef, label %bb122, label %bb121

bb121:                                            ; preds = %bb110
  br label %bb122

bb122:                                            ; preds = %bb121, %bb110
  br label %bb110

bb154:                                            ; preds = %bb150.preheader
  unreachable

bb160:                                            ; preds = %bb150.preheader
  unreachable

return:                                           ; preds = %bb18, %entry
  ret void
}
