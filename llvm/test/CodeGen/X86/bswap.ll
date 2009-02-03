; bswap should be constant folded when it is passed a constant argument

; RUN: llvm-as < %s | llc -march=x86 | \
; RUN:   grep bswapl | count 3
; RUN: llvm-as < %s | llc -march=x86 | grep rolw | count 1

declare i16 @llvm.bswap.i16(i16)

declare i32 @llvm.bswap.i32(i32)

declare i64 @llvm.bswap.i64(i64)

define i16 @W(i16 %A) {
        %Z = call i16 @llvm.bswap.i16( i16 %A )         ; <i16> [#uses=1]
        ret i16 %Z
}

define i32 @X(i32 %A) {
        %Z = call i32 @llvm.bswap.i32( i32 %A )         ; <i32> [#uses=1]
        ret i32 %Z
}

define i64 @Y(i64 %A) {
        %Z = call i64 @llvm.bswap.i64( i64 %A )         ; <i64> [#uses=1]
        ret i64 %Z
}

