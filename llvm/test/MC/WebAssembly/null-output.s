# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj -o /dev/null < %s

    .text
    .section .text.main,"",@
    .type    main,@function
main:
    .functype   main (i32, i32) -> (i32)
    local.get 0
    end_function
.Lfunc_end0:
    .size main, .Lfunc_end0-main
