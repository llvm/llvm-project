; These tests should not contain a sign extend.
; RUN: llc -verify-machineinstrs < %s -march=ppc32 | not grep extsh
; RUN: llc -verify-machineinstrs < %s -march=ppc32 | not grep extsb

define i32 @test1(i32 %mode.0.i.0) {
        %tmp.79 = trunc i32 %mode.0.i.0 to i16
        %tmp.80 = sext i16 %tmp.79 to i32
        %tmp.81 = and i32 %tmp.80, 24
        ret i32 %tmp.81
}

define signext i16 @test2(i16 signext %X, i16 signext %x)  {
        %tmp = sext i16 %X to i32
        %tmp1 = sext i16 %x to i32
        %tmp2 = add i32 %tmp, %tmp1
        %tmp4 = ashr i32 %tmp2, 1
        %tmp5 = trunc i32 %tmp4 to i16
        %tmp45 = sext i16 %tmp5 to i32
        %retval = trunc i32 %tmp45 to i16
        ret i16 %retval
}

define signext i16 @test3(i32 zeroext %X)  {
        %tmp1 = lshr i32 %X, 16
        %tmp2 = trunc i32 %tmp1 to i16
        ret i16 %tmp2
}

