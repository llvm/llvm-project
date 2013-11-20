// RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux %s -o -| llvm-readobj -t | FileCheck %s

        .text
        .globl  main
        .align  2
        .type   main,@function
        .set    nomips16                # @main
        .ent    main
        .mips_hack_stocg main, 128
main:

// CHECK:     Name: main
// CHECK:     Other: 128
