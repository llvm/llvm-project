; RUN: %lli -mtriple=%mcjit_triple -use-mcjit -force-interpreter=true %s > /dev/null

define i32 @main() {
       %a = add i32 0, undef
       %b = fadd float 0.0, undef
       %c = fadd double 0.0, undef
       ret i32 0
}
