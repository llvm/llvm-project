// RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o /dev/null 2>%t
// RUN: FileCheck --input-file=%t %s


        .data
x:
// CHECK: [[@LINE+1]]:{{[0-9]+}}: error: symbol '__executable_start' can not be undefined in a subtraction expression
        .quad   x-__executable_start

// CHECK: [[@LINE+1]]:{{[0-9]+}}: error: unsupported subtraction of qualified symbol
        .long bar - foo@got
foo:
