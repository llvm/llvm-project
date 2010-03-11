; RUN: llc -march=x86 %s -o -
; RUN: llc -march=x86-64 %s -o -

; PR6497

; Chain and flag folding issues.
define i32 @test1() nounwind ssp {
entry:
  %tmp5.i = volatile load i32* undef              ; <i32> [#uses=1]
  %conv.i = zext i32 %tmp5.i to i64               ; <i64> [#uses=1]
  %tmp12.i = volatile load i32* undef             ; <i32> [#uses=1]
  %conv13.i = zext i32 %tmp12.i to i64            ; <i64> [#uses=1]
  %shl.i = shl i64 %conv13.i, 32                  ; <i64> [#uses=1]
  %or.i = or i64 %shl.i, %conv.i                  ; <i64> [#uses=1]
  %add16.i = add i64 %or.i, 256                   ; <i64> [#uses=1]
  %shr.i = lshr i64 %add16.i, 8                   ; <i64> [#uses=1]
  %conv19.i = trunc i64 %shr.i to i32             ; <i32> [#uses=1]
  volatile store i32 %conv19.i, i32* undef
  ret i32 undef
}

; PR6533
define void @test2(i1 %x, i32 %y) nounwind {
  %land.ext = zext i1 %x to i32                   ; <i32> [#uses=1]
  %and = and i32 %y, 1                        ; <i32> [#uses=1]
  %xor = xor i32 %and, %land.ext                  ; <i32> [#uses=1]
  %cmp = icmp eq i32 %xor, 1                      ; <i1> [#uses=1]
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %land.end
  ret void

if.end:                                           ; preds = %land.end
  ret void
}
