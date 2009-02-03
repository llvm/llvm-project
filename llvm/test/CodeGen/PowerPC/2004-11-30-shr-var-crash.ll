; RUN: llvm-as < %s | llc -march=ppc32

define void @main() {
        %shamt = add i8 0, 1            ; <i8> [#uses=1]
        %shift.upgrd.1 = zext i8 %shamt to i64          ; <i64> [#uses=1]
        %tr2 = ashr i64 1, %shift.upgrd.1               ; <i64> [#uses=0]
        ret void
}

