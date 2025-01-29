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
