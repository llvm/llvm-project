; RUN: llc < %s -mtriple=i686--
; PR4753

; This function has a sub-register reuse undone.

@uint8 = external dso_local global i32                      ; <i32*> [#uses=3]

declare signext i8 @foo(i32, i8 signext) nounwind readnone

declare signext i8 @bar(i32, i8 signext) nounwind readnone

define i32 @uint80(i8 signext %p_52) nounwind {
entry:
  %0 = sext i8 %p_52 to i16                       ; <i16> [#uses=1]
  %1 = tail call i32 @func_24(i16 zeroext %0, i8 signext ptrtoint (i8 (i32, i8)* @foo to i8)) nounwind; <i32> [#uses=1]
  %2 = trunc i32 %1 to i8                         ; <i8> [#uses=1]
  %3 = or i8 %2, 1                                ; <i8> [#uses=1]
  %4 = tail call i32 @safe(i32 1) nounwind        ; <i32> [#uses=0]
  %5 = tail call i32 @func_24(i16 zeroext 0, i8 signext undef) nounwind; <i32> [#uses=1]
  %6 = trunc i32 %5 to i8                         ; <i8> [#uses=1]
  %7 = xor i8 %3, %p_52                           ; <i8> [#uses=1]
  %8 = xor i8 %7, %6                              ; <i8> [#uses=1]
  %9 = icmp ne i8 %p_52, 0                        ; <i1> [#uses=1]
  %10 = zext i1 %9 to i8                          ; <i8> [#uses=1]
  %11 = tail call i32 @func_24(i16 zeroext ptrtoint (i8 (i32, i8)* @bar to i16), i8 signext %10) nounwind; <i32> [#uses=1]
  %12 = tail call i32 @func_24(i16 zeroext 0, i8 signext 1) nounwind; <i32> [#uses=0]
  br i1 undef, label %bb2, label %bb

bb:                                               ; preds = %entry
  br i1 undef, label %bb2, label %bb3

bb2:                                              ; preds = %bb, %entry
  br label %bb3

bb3:                                              ; preds = %bb2, %bb
  %iftmp.2.0 = phi i32 [ 0, %bb2 ], [ 1, %bb ]    ; <i32> [#uses=1]
  %13 = icmp ne i32 %11, %iftmp.2.0               ; <i1> [#uses=1]
  %14 = tail call i32 @safe(i32 -2) nounwind      ; <i32> [#uses=0]
  %15 = zext i1 %13 to i8                         ; <i8> [#uses=1]
  %16 = tail call signext i8 @func_53(i8 signext undef, i8 signext 1, i8 signext %15, i8 signext %8) nounwind; <i8> [#uses=0]
  br i1 undef, label %bb5, label %bb4

bb4:                                              ; preds = %bb3
  %17 = load volatile i32, i32* @uint8, align 4        ; <i32> [#uses=0]
  br label %bb5

bb5:                                              ; preds = %bb4, %bb3
  %18 = load volatile i32, i32* @uint8, align 4        ; <i32> [#uses=0]
  %19 = sext i8 undef to i16                      ; <i16> [#uses=1]
  %20 = tail call i32 @func_24(i16 zeroext %19, i8 signext 1) nounwind; <i32> [#uses=0]
  br i1 undef, label %return, label %bb6.preheader

bb6.preheader:                                    ; preds = %bb5
  %21 = sext i8 %p_52 to i32                      ; <i32> [#uses=1]
  %22 = load volatile i32, i32* @uint8, align 4        ; <i32> [#uses=0]
  %23 = tail call i32 (...) @safefuncts(i32 %21, i32 1) nounwind; <i32> [#uses=0]
  unreachable

return:                                           ; preds = %bb5
  ret i32 undef
}

declare i32 @func_24(i16 zeroext, i8 signext)

declare i32 @safe(i32)

declare signext i8 @func_53(i8 signext, i8 signext, i8 signext, i8 signext)

declare i32 @safefuncts(...)
