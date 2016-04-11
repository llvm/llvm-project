// RUN: not llvm-mc -triple x86_64-apple-macosx -filetype=obj -o /dev/null %s 2>&1 | FileCheck  %s

// CHECK: [[@LINE+1]]:{{[0-9]+}}: error: directional label undefined
        jmp 1b
// CHECK: [[@LINE+1]]:{{[0-9]+}}: error: directional label undefined
        jmp 1f
// CHECK: [[@LINE+1]]:{{[0-9]+}}: error: directional label undefined
        jmp 2f

