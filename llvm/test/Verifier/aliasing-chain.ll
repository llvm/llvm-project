; RUN:  not llvm-as -f %s -o /dev/null |& grep {Aliasing chain should end with function or global variable}

; Test that alising chain does not create a cycle

@b1 = alias i32* @c1
@c1 = alias i32* @b1
