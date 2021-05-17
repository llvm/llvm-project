; RUN: %lli -jit-kind=mcjit -force-interpreter=true %s

define i32 @main() {
       %a = add i32 0, undef
       %b = fadd float 0.0, undef
       %c = fadd double 0.0, undef
       ret i32 0
}
