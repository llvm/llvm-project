; RUN: llvm-as < %s | llc -march=x86 -regalloc=simple

define i32 @main(i32 %B) {
        ;%B = add i32 0, 1;
        %R = sub i32 %B, 1 ; %r = 0
        ret i32 %R
}
