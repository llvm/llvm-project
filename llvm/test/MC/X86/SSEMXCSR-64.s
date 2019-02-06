// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: ldmxcsr 485498096
// CHECK: encoding: [0x0f,0xae,0x14,0x25,0xf0,0x1c,0xf0,0x1c]
ldmxcsr 485498096

// CHECK: ldmxcsr 64(%rdx)
// CHECK: encoding: [0x0f,0xae,0x52,0x40]
ldmxcsr 64(%rdx)

// CHECK: ldmxcsr -64(%rdx,%rax,4)
// CHECK: encoding: [0x0f,0xae,0x54,0x82,0xc0]
ldmxcsr -64(%rdx,%rax,4)

// CHECK: ldmxcsr 64(%rdx,%rax,4)
// CHECK: encoding: [0x0f,0xae,0x54,0x82,0x40]
ldmxcsr 64(%rdx,%rax,4)

// CHECK: ldmxcsr 64(%rdx,%rax)
// CHECK: encoding: [0x0f,0xae,0x54,0x02,0x40]
ldmxcsr 64(%rdx,%rax)

// CHECK: ldmxcsr (%rdx)
// CHECK: encoding: [0x0f,0xae,0x12]
ldmxcsr (%rdx)

// CHECK: stmxcsr 485498096
// CHECK: encoding: [0x0f,0xae,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]
stmxcsr 485498096

// CHECK: stmxcsr 64(%rdx)
// CHECK: encoding: [0x0f,0xae,0x5a,0x40]
stmxcsr 64(%rdx)

// CHECK: stmxcsr -64(%rdx,%rax,4)
// CHECK: encoding: [0x0f,0xae,0x5c,0x82,0xc0]
stmxcsr -64(%rdx,%rax,4)

// CHECK: stmxcsr 64(%rdx,%rax,4)
// CHECK: encoding: [0x0f,0xae,0x5c,0x82,0x40]
stmxcsr 64(%rdx,%rax,4)

// CHECK: stmxcsr 64(%rdx,%rax)
// CHECK: encoding: [0x0f,0xae,0x5c,0x02,0x40]
stmxcsr 64(%rdx,%rax)

// CHECK: stmxcsr (%rdx)
// CHECK: encoding: [0x0f,0xae,0x1a]
stmxcsr (%rdx)

