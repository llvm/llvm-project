; RUN: llc < %s -march=msp430 | grep rra | count 1

define i16 @lsr2u16(i16 %x.arg) nounwind {
        %retval = alloca i16
        %x = alloca i16
        store i16 %x.arg, ptr %x
        %1 = load i16, ptr %x
        %2 = lshr i16 %1, 2
        store i16 %2, ptr %retval
        br label %return
return:
        %3 = load i16, ptr %retval
        ret i16 %3

}
