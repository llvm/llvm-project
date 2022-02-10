; RUN: opt < %s -passes=instcombine -disable-output
; PR1594

define i64 @test(i16 %tmp510, i16 %tmp512) {
	%W = sext i16 %tmp510 to i32           ; <i32> [#uses=1]
        %X = sext i16 %tmp512 to i32           ; <i32> [#uses=1]
        %Y = add i32 %W, %X               ; <i32> [#uses=1]
        %Z = sext i32 %Y to i64          ; <i64> [#uses=1]
	ret i64 %Z
}
