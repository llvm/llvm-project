; LoongArch does not support mcjit.
; UNSUPPORTED: target=loongarch{{.*}}

; RUN: %lli -jit-kind=mcjit %s > /dev/null
; RUN: %lli %s > /dev/null

define i32 @main() {
    %A_i8 = add <5 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4>, <i8 12, i8 34, i8 56, i8 78, i8 89>
    %B_i8 = sub <5 x i8> %A_i8, <i8 11, i8 22, i8 33, i8 44, i8 55>
    %C_i8 = mul <5 x i8> %B_i8, %B_i8
    %D_i8 = sdiv <5 x i8> %C_i8, %C_i8
    %E_i8 = srem <5 x i8> %D_i8, %D_i8
    %F_i8 = udiv <5 x i8> <i8 5, i8 6, i8 7, i8 8, i8 9>, <i8 6, i8 5, i8 4, i8 3, i8 2>
    %G_i8 = urem <5 x i8> <i8 6, i8 7, i8 8, i8 9, i8 10>, <i8 5, i8 4, i8 2, i8 2, i8 1>

    %A_i16 = add <4 x i16> <i16 0, i16 1, i16 2, i16 3>, <i16 123, i16 345, i16 567, i16 789>
    %B_i16 = sub <4 x i16> %A_i16, <i16 111, i16 222, i16 333, i16 444>
    %C_i16 = mul <4 x i16> %B_i16, %B_i16
    %D_i16 = sdiv <4 x i16> %C_i16, %C_i16
    %E_i16 = srem <4 x i16> %D_i16, %D_i16
    %F_i16 = udiv <4 x i16> <i16 5, i16 6, i16 7, i16 8>, <i16 6, i16 5, i16 4, i16 3>
    %G_i16 = urem <4 x i16> <i16 6, i16 7, i16 8, i16 9>, <i16 5, i16 4, i16 3, i16 2>

    %A_i32 = add <3 x i32> <i32 0, i32 1, i32 2>, <i32 1234, i32 3456, i32 5678>
    %B_i32 = sub <3 x i32> %A_i32, <i32 1111, i32 2222, i32 3333>
    %C_i32 = mul <3 x i32> %B_i32, %B_i32
    %D_i32 = sdiv <3 x i32> %C_i32, %C_i32
    %E_i32 = srem <3 x i32> %D_i32, %D_i32
    %F_i32 = udiv <3 x i32> <i32 5, i32 6, i32 7>, <i32 6, i32 5, i32 4>
    %G_i32 = urem <3 x i32> <i32 6, i32 7, i32 8>, <i32 5, i32 4, i32 3>

    %A_i64 = add <2 x i64> <i64 0, i64 1>, <i64 12455, i64 34567>
    %B_i64 = sub <2 x i64> %A_i64, <i64 11111, i64 22222>
    %C_i64 = mul <2 x i64> %B_i64, %B_i64
    %D_i64 = sdiv <2 x i64> %C_i64, %C_i64
    %E_i64 = srem <2 x i64> %D_i64, %D_i64
    %F_i64 = udiv <2 x i64> <i64 5, i64 6>, <i64 6, i64 5>
    %G_i64 = urem <2 x i64> <i64 6, i64 7>, <i64 5, i64 3>
 
    ret i32 0
}
