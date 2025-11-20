; RUN: %lli -jit-kind=mcjit -force-interpreter=true %s

define i32 @main() {
       %a = add i32 0, undef
       %b = fadd float 0.0, undef
       %c = fadd double 0.0, undef
       %d = insertvalue {i32, [4 x i32]} undef, i32 1, 1, 2
       ret i32 0
}
