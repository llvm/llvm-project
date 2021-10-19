; RUN: %lli -jit-kind=mcjit %s > /dev/null
; RUN: %lli %s > /dev/null

define i32 @main() {
    %A_i8 = and <5 x i8> <i8 4, i8 4, i8 4, i8 4, i8 4>, <i8 8, i8 8, i8 8, i8 8, i8 8>
    %B_i8 = or <5 x i8> %A_i8, <i8 7, i8 7, i8 7, i8 7, i8 7>
    %C_i8 = xor <5 x i8> %B_i8, %A_i8

    %A_i16 = and <4 x i16> <i16 4, i16 4, i16 4, i16 4>, <i16 8, i16 8, i16 8, i16 8>
    %B_i16 = or <4 x i16> %A_i16, <i16 7, i16 7, i16 7, i16 7>
    %C_i16 = xor <4 x i16> %B_i16, %A_i16

    %A_i32 = and <3 x i32> <i32 4, i32 4, i32 4>, <i32 8, i32 8, i32 8>
    %B_i32 = or <3 x i32> %A_i32, <i32 7, i32 7, i32 7>
    %C_i32 = xor <3 x i32> %B_i32, %A_i32

    %A_i64 = and <2 x i64> <i64 4, i64 4>, <i64 8, i64 8>
    %B_i64 = or <2 x i64> %A_i64, <i64 7, i64 7>
    %C_i64 = xor <2 x i64> %B_i64, %A_i64

    ret i32 0
}

