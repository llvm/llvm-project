// RUN: not llvm-mc -triple x86_64-linux-gnu -filetype=obj -o /dev/null %s 2>&1 | FileCheck  %s

// CHECK: [[@LINE+1]]:{{[0-9]+}}: error: directional label undefined
        jmp 1b
// CHECK: [[@LINE+1]]:{{[0-9]+}}: error: directional label undefined
        jmp 1f
# 10 "wibble.s"
// CHECK: wibble.s:11:{{[0-9]+}}: error: directional label undefined
        jmp 2f

# 42 "invalid.s"

