; RUN: llvm-as < %s | llc -march=x86 | grep {sar.	\$5}

@foo = global i8 127

define i32 @main() nounwind {
entry:
        %tmp = load i8* @foo
        %bf.lo = lshr i8 %tmp, 5
        %bf.lo.cleared = and i8 %bf.lo, 7
        %0 = shl i8 %bf.lo.cleared, 5
        %bf.val.sext = ashr i8 %0, 5
        %conv = sext i8 %bf.val.sext to i32
        ret i32 %conv
}
