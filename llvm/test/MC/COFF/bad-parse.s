// RUN: not llvm-mc -filetype=obj -triple i386-pc-win32 %s 2>&1 | FileCheck %s

        .data

// CHECK: [[@LINE+1]]:{{[0-9]+}}: error: expected identifier in directive
        .secnum
// CHECK: [[@LINE+1]]:{{[0-9]+}}: error: unexpected token in directive
        .secnum section extra

// CHECK: [[@LINE+1]]:{{[0-9]+}}: error: expected identifier in directive
        .secoffset
// CHECK: [[@LINE+1]]:{{[0-9]+}}: error: unexpected token in directive
        .secoffset section extra

// CHECK: [[@LINE+1]]:{{[0-9]+}}: error: expected unwind version number
        .seh_unwindversion
// CHECK: [[@LINE+1]]:{{[0-9]+}}: error: expected unwind version number
        .seh_unwindversion hello
// CHECK: [[@LINE+1]]:{{[0-9]+}}: error: invalid unwind version
        .seh_unwindversion 0
// CHECK: [[@LINE+1]]:{{[0-9]+}}: error: invalid unwind version
        .seh_unwindversion 9000
// CHECK: [[@LINE+1]]:{{[0-9]+}}: error: unexpected token in directive
        .seh_unwindversion 2 hello
