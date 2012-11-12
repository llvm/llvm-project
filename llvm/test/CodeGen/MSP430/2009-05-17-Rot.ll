; RUN: llc < %s -march=msp430

define i16 @rol1u16(i16 %x.arg) nounwind {
        %retval = alloca i16
        %x = alloca i16
        store i16 %x.arg, i16* %x
        %1 = load i16* %x
        %2 = shl i16 %1, 1
        %3 = load i16* %x
        %4 = lshr i16 %3, 15
        %5 = or i16 %2, %4
        store i16 %5, i16* %retval
        br label %return
return:
        %6 = load i16* %retval
        ret i16 %6
}
