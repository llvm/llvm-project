; bswap should be constant folded when it is passed a constant argument

; RUN: opt < %s -passes=instsimplify -S | FileCheck %s

declare i16 @llvm.bswap.i16(i16)

declare i32 @llvm.bswap.i32(i32)

declare i64 @llvm.bswap.i64(i64)

declare i80 @llvm.bswap.i80(i80)

; CHECK-LABEL: define i16 @W(
define i16 @W() {
        ; CHECK: ret i16 256
        %Z = call i16 @llvm.bswap.i16( i16 1 )          ; <i16> [#uses=1]
        ret i16 %Z
}

; CHECK-LABEL: define i32 @X(
define i32 @X() {
        ; CHECK: ret i32 16777216
        %Z = call i32 @llvm.bswap.i32( i32 1 )          ; <i32> [#uses=1]
        ret i32 %Z
}

; CHECK-LABEL: define i64 @Y(
define i64 @Y() {
        ; CHECK: ret i64 72057594037927936
        %Z = call i64 @llvm.bswap.i64( i64 1 )          ; <i64> [#uses=1]
        ret i64 %Z
}

; CHECK-LABEL: define i80 @Z(
define i80 @Z() {
        ; CHECK: ret i80 -450681596205739728166896
        ;                0xA0908070605040302010
        %Z = call i80 @llvm.bswap.i80( i80 76151636403560493650080 )
        ;                                  0x102030405060708090A0
        ret i80 %Z
}
