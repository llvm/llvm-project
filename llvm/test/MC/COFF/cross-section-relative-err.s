// RUN: not llvm-mc -filetype=obj -triple i686-pc-win32 %s -o /dev/null 2>&1 | FileCheck %s --check-prefixes=CHECK,I686
// RUN: not llvm-mc -filetype=obj -triple x86_64-pc-win32 %s -o /dev/null 2>&1 | FileCheck %s


// CHECK: [[@LINE+1]]:{{[0-9]+}}: error: Cannot represent this expression
        .byte foo - .

// CHECK: [[@LINE+1]]:{{[0-9]+}}: error: Cannot represent this expression
        .short foo - .

// I686: [[@LINE+1]]:{{[0-9]+}}: error: Cannot represent this expression
        .quad foo - .
