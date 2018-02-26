; RUN: llc < %s -mtriple=i686-- | grep "movsbl"

@X = global i32 0               ; <i32*> [#uses=1]

define i32 @_Z3fooi(i32 %x)   {
entry:
        store i32 %x, i32* @X, align 4
        %retval67 = trunc i32 %x to i8          ; <i8> [#uses=1]
        %retval = sext i8 %retval67 to i32
        ret i32 %retval
}
