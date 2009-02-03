; RUN: llvm-as < %s | llc -mtriple=x86_64-unknown-linux-gnu
; PR2289

define void @_ada_ca11001() {
entry:
        %tmp59 = call i16 @ca11001_0__cartesian_assign( i8 zeroext  0, i8 zeroext  0, i16 undef )               ; <i16> [#uses=0]
        unreachable
}

declare i16 @ca11001_0__cartesian_assign(i8 zeroext , i8 zeroext , i16)
