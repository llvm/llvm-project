; RUN: llvm-as < %s | opt -globalopt -disable-output
; PR579

@g_40507551 = internal global i16 31038         ; <i16*> [#uses=1]

define void @main() {
        %tmp.4.i.1 = load i8* getelementptr (i8* bitcast (i16* @g_40507551 to i8*), i32 1)              ; <i8> [#uses=0]
        ret void
}

