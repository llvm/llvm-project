; LoongArch does not support mcjit.
; UNSUPPORTED: target=loongarch{{.*}}

; RUN: %lli -jit-kind=mcjit %s > /dev/null
; RUN: %lli %s > /dev/null

define i32 @main() {
    %int1 = add <3 x i32> <i32 0, i32 0, i32 0>, <i32 0, i32 0, i32 0>
    %int2 = add <3 x i32> <i32 0, i32 0, i32 0>, <i32 0, i32 0, i32 0>
    %long1 = add <2 x i64> <i64 0, i64 0>, <i64 0, i64 0>
    %long2 = add <2 x i64> <i64 0, i64 0>, <i64 0, i64 0>
    %sbyte1 = add <5 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0>, <i8 0, i8 0, i8 0, i8 0, i8 0>
    %sbyte2 = add <5 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0>, <i8 0, i8 0, i8 0, i8 0, i8 0>
    %short1 = add <4 x i16> <i16 0, i16 0, i16 0, i16 0>, <i16 0, i16 0, i16 0, i16 0>
    %short2 = add <4 x i16> <i16 0, i16 0, i16 0, i16 0>, <i16 0, i16 0, i16 0, i16 0>
    %ubyte1 = add <5 x i8>  <i8 0, i8 0, i8 0, i8 0, i8 0>, <i8 0, i8 0, i8 0, i8 0, i8 0>
    %ubyte2 = add <5 x i8>  <i8 0, i8 0, i8 0, i8 0, i8 0>, <i8 0, i8 0, i8 0, i8 0, i8 0>
    %uint1 = add <3 x i32> <i32 0, i32 0, i32 0>, <i32 0, i32 0, i32 0>
    %uint2 = add <3 x i32> <i32 0, i32 0, i32 0>, <i32 0, i32 0, i32 0>
    %ulong1 = add <2 x i64> <i64 0, i64 0>, <i64 0, i64 0>
    %ulong2 = add <2 x i64> <i64 0, i64 0>, <i64 0, i64 0>
    %ushort1 = add <4 x i16> <i16 0, i16 0, i16 0, i16 0>, <i16 0, i16 0, i16 0, i16 0>
    %ushort2 = add <4 x i16> <i16 0, i16 0, i16 0, i16 0>, <i16 0, i16 0, i16 0, i16 0>
    %test1 = icmp eq <5 x i8> %ubyte1, %ubyte2
    %test2 = icmp uge <5 x i8> %ubyte1, %ubyte2
    %test3 = icmp ugt <5 x i8> %ubyte1, %ubyte2
    %test4 = icmp ule <5 x i8> %ubyte1, %ubyte2
    %test5 = icmp ult <5 x i8> %ubyte1, %ubyte2
    %test6 = icmp ne <5 x i8> %ubyte1, %ubyte2
    %test7 = icmp eq <4 x i16> %ushort1, %ushort2
    %test8 = icmp uge <4 x i16> %ushort1, %ushort2
    %test9 = icmp ugt <4 x i16> %ushort1, %ushort2
    %test10 = icmp ule <4 x i16> %ushort1, %ushort2
    %test11 = icmp ult <4 x i16> %ushort1, %ushort2
    %test12 = icmp ne <4 x i16> %ushort1, %ushort2 
    %test13 = icmp eq <3 x i32> %uint1, %uint2
    %test14 = icmp uge <3 x i32> %uint1, %uint2
    %test15 = icmp ugt <3 x i32> %uint1, %uint2
    %test16 = icmp ule <3 x i32> %uint1, %uint2
    %test17 = icmp ult <3 x i32> %uint1, %uint2
    %test18 = icmp ne <3 x i32> %uint1, %uint2
    %test19 = icmp eq <2 x i64> %ulong1, %ulong2
    %test20 = icmp uge <2 x i64> %ulong1, %ulong2
    %test21 = icmp ugt <2 x i64> %ulong1, %ulong2
    %test22 = icmp ule <2 x i64> %ulong1, %ulong2
    %test23 = icmp ult <2 x i64> %ulong1, %ulong2
    %test24 = icmp ne <2 x i64> %ulong1, %ulong2
    %test25 = icmp eq <5 x i8> %sbyte1, %sbyte2
    %test26 = icmp sge <5 x i8> %sbyte1, %sbyte2
    %test27 = icmp sgt <5 x i8> %sbyte1, %sbyte2
    %test28 = icmp sle <5 x i8> %sbyte1, %sbyte2
    %test29 = icmp slt <5 x i8> %sbyte1, %sbyte2
    %test30 = icmp ne <5 x i8> %sbyte1, %sbyte2
    %test31 = icmp eq <4 x i16> %short1, %short2
    %test32 = icmp sge <4 x i16> %short1, %short2
    %test33 = icmp sgt <4 x i16> %short1, %short2
    %test34 = icmp sle <4 x i16> %short1, %short2
    %test35 = icmp slt <4 x i16> %short1, %short2
    %test36 = icmp ne <4 x i16> %short1, %short2
    %test37 = icmp eq <3 x i32> %int1, %int2
    %test38 = icmp sge <3 x i32> %int1, %int2
    %test39 = icmp sgt <3 x i32> %int1, %int2
    %test40 = icmp sle <3 x i32> %int1, %int2
    %test41 = icmp slt <3 x i32> %int1, %int2
    %test42 = icmp ne <3 x i32> %int1, %int2
    %test43 = icmp eq <2 x i64> %long1, %long2
    %test44 = icmp sge <2 x i64> %long1, %long2
    %test45 = icmp sgt <2 x i64> %long1, %long2
    %test46 = icmp sle <2 x i64> %long1, %long2
    %test47 = icmp slt <2 x i64> %long1, %long2
    %test48 = icmp ne <2 x i64> %long1, %long2
    ret i32 0
}
