// RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t 2> %t.out
// RUN: FileCheck --input-file=%t.out %s

// CHECK: A @@ version cannot be undefined

        .symver undefined, foo@@bar
        .long undefined
