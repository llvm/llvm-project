; RUN: %lli -jit-kind=mcjit -force-interpreter=true %s > /dev/null

define i32 @main() {
    %shamt = add <2 x i8> <i8 0, i8 0>, <i8 1, i8 2>
    %shift.upgrd.1 = zext <2 x i8> %shamt to <2 x i32>
    %t1.s = shl <2 x i32> <i32 1, i32 2>, %shift.upgrd.1
    %t2.s = shl <2 x i32> <i32 1, i32 2>, <i32 3, i32 4>
    %shift.upgrd.2 = zext <2 x i8> %shamt to <2 x i32>
    %t1 = shl <2 x i32> <i32 1, i32 2>, %shift.upgrd.2
    %t2 = shl <2 x i32> <i32 1, i32 0>, <i32 5, i32 6>
    %t2.s.upgrd.3 = shl <2 x i64> <i64 1, i64 2>, <i64 3, i64 4>
    %t2.upgrd.4 = shl <2 x i64> <i64 1, i64 2>, <i64 6, i64 7>
    %shift.upgrd.5 = zext <2 x i8> %shamt to <2 x i32>
    %tr1.s = ashr <2 x i32> <i32 1, i32 2>, %shift.upgrd.5
    %tr2.s = ashr <2 x i32> <i32 1, i32 2>, <i32 4, i32 5>
    %shift.upgrd.6 = zext <2 x i8> %shamt to <2 x i32>
    %tr1 = lshr <2 x i32> <i32 1, i32 2>, %shift.upgrd.6
    %tr2 = lshr <2 x i32> <i32 1, i32 2>, <i32 5, i32 6>
    %tr1.l = ashr <2 x i64> <i64 1, i64 2>, <i64 4, i64 5>
    %shift.upgrd.7 = zext <2 x i8> %shamt to <2 x i64>
    %tr2.l = ashr <2 x i64> <i64 1, i64 2>, %shift.upgrd.7
    %tr3.l = shl <2 x i64> <i64 1, i64 2>, <i64 4, i64 5>
    %shift.upgrd.8 = zext <2 x i8> %shamt to <2 x i64>
    %tr4.l = shl <2 x i64> <i64 1, i64 2>, %shift.upgrd.8
    %tr1.u = lshr <2 x i64> <i64 1, i64 2>, <i64 5, i64 6>
    %shift.upgrd.9 = zext <2 x i8> %shamt to <2 x i64>
    %tr2.u = lshr <2 x i64> <i64 1, i64 2>, %shift.upgrd.9
    %tr3.u = shl <2 x i64> <i64 1, i64 2>, <i64 5, i64 6>
    %shift.upgrd.10 = zext <2 x i8> %shamt to <2 x i64>
    %tr4.u = shl <2 x i64> <i64 1, i64 2>, %shift.upgrd.10
    ret i32 0
}
