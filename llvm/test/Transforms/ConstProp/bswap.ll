; bswap should be constant folded when it is passed a constant argument

; RUN: llvm-as < %s | opt -constprop | llvm-dis | not grep call

declare i16 @llvm.bswap.i16(i16)

declare i32 @llvm.bswap.i32(i32)

declare i64 @llvm.bswap.i64(i64)

define i16 @W() {
        %Z = call i16 @llvm.bswap.i16( i16 1 )          ; <i16> [#uses=1]
        ret i16 %Z
}

define i32 @X() {
        %Z = call i32 @llvm.bswap.i32( i32 1 )          ; <i32> [#uses=1]
        ret i32 %Z
}

define i64 @Y() {
        %Z = call i64 @llvm.bswap.i64( i64 1 )          ; <i64> [#uses=1]
        ret i64 %Z
}

