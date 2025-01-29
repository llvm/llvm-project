// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: prefetchrst2 485498096
// CHECK: encoding: [0x0f,0x18,0x24,0x25,0xf0,0x1c,0xf0,0x1c]
prefetchrst2 485498096

// CHECK: prefetchrst2 64(%rdx)
// CHECK: encoding: [0x0f,0x18,0x62,0x40]
prefetchrst2 64(%rdx)

// CHECK: prefetchrst2 64(%rdx,%rax,4)
// CHECK: encoding: [0x0f,0x18,0x64,0x82,0x40]
prefetchrst2 64(%rdx,%rax,4)

// CHECK: prefetchrst2 -64(%rdx,%rax,4)
// CHECK: encoding: [0x0f,0x18,0x64,0x82,0xc0]
prefetchrst2 -64(%rdx,%rax,4)

// CHECK: prefetchrst2 64(%rdx,%rax)
// CHECK: encoding: [0x0f,0x18,0x64,0x02,0x40]
prefetchrst2 64(%rdx,%rax)

// CHECK: prefetchrst2 (%rdx)
// CHECK: encoding: [0x0f,0x18,0x22]
prefetchrst2 (%rdx)