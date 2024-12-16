// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: prefetchrst2 -485498096(%edx,%eax,4)
// CHECK: encoding: [0x0f,0x18,0xa4,0x82,0x10,0xe3,0x0f,0xe3]
prefetchrst2 -485498096(%edx,%eax,4)

// CHECK: prefetchrst2 485498096(%edx,%eax,4)
// CHECK: encoding: [0x0f,0x18,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]
prefetchrst2 485498096(%edx,%eax,4)

// CHECK: prefetchrst2 485498096(%edx)
// CHECK: encoding: [0x0f,0x18,0xa2,0xf0,0x1c,0xf0,0x1c]
prefetchrst2 485498096(%edx)

// CHECK: prefetchrst2 485498096
// CHECK: encoding: [0x0f,0x18,0x25,0xf0,0x1c,0xf0,0x1c]
prefetchrst2 485498096

// CHECK: prefetchrst2 64(%edx,%eax)
// CHECK: encoding: [0x0f,0x18,0x64,0x02,0x40]
prefetchrst2 64(%edx,%eax)

// CHECK: prefetchrst2 (%edx)
// CHECK: encoding: [0x0f,0x18,0x22]
prefetchrst2 (%edx)