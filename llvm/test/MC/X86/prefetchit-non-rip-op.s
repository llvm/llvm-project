// RUN: llvm-mc -triple x86_64-unknown-unknown %s > %t 2> %t.err
// RUN: FileCheck --check-prefix=CHECK-STDERR < %t.err %s
// RUN: FileCheck < %t %s

// CHECK: prefetchit0 (%rdi)
// CHECK-STDERR: warning: 'prefetchit0' only supports RIP-relative address
          prefetchit0 (%rdi)

// CHECK: prefetchit1 (%rcx)
// CHECK-STDERR: warning: 'prefetchit1' only supports RIP-relative address
          prefetchit1 (%rcx)

// CHECK: prefetchit0 1(%rdx)
// CHECK-STDERR: warning: 'prefetchit0' only supports RIP-relative address
          prefetchit0 1(%rdx)

// CHECK: prefetchit1 12(%rsi)
// CHECK-STDERR: warning: 'prefetchit1' only supports RIP-relative address
          prefetchit1 12(%rsi)

// CHECK: prefetchit0 123(%r8,%rax)
// CHECK-STDERR: warning: 'prefetchit0' only supports RIP-relative address
          prefetchit0 123(%r8,%rax)

// CHECK: prefetchit1 1234(%r9,%r10)
// CHECK-STDERR: warning: 'prefetchit1' only supports RIP-relative address
          prefetchit1 1234(%r9,%r10)

// CHECK: prefetchit0 (%r11,%r12)
// CHECK-STDERR: warning: 'prefetchit0' only supports RIP-relative address
          prefetchit0 (%r11,%r12)

// CHECK: prefetchit1 (%r13,%r14)
// CHECK-STDERR: warning: 'prefetchit1' only supports RIP-relative address
          prefetchit1 (%r13,%r14)

// CHECK: prefetchit0 987(%rsp,%r15,4)
// CHECK-STDERR: warning: 'prefetchit0' only supports RIP-relative address
          prefetchit0 987(%rsp,%r15,4)

// CHECK: prefetchit1 -1(%rbp,%rdi,8)
// CHECK-STDERR: warning: 'prefetchit1' only supports RIP-relative address
          prefetchit1 -1(%rbp,%rdi,8)

// CHECK: prefetchit0 (%rsp,%rsi,2)
// CHECK-STDERR: warning: 'prefetchit0' only supports RIP-relative address
          prefetchit0 (%rsp,%rsi,2)

// CHECK: prefetchit1 (%rdi,%r15,4)
// CHECK-STDERR: warning: 'prefetchit1' only supports RIP-relative address
          prefetchit1 (%rdi,%r15,4)

// CHECK: prefetchit0 80(,%r14,8)
// CHECK-STDERR: warning: 'prefetchit0' only supports RIP-relative address
          prefetchit0 80(,%r14,8)

// CHECK: prefetchit1 3(,%r8,4)
// CHECK-STDERR: warning: 'prefetchit1' only supports RIP-relative address
          prefetchit1 3(,%r8,4)

// CHECK: prefetchit0 (,%rax,2)
// CHECK-STDERR: warning: 'prefetchit0' only supports RIP-relative address
          prefetchit0 (,%rax,2)

// CHECK: prefetchit1 (,%rcx,8)
// CHECK-STDERR: warning: 'prefetchit1' only supports RIP-relative address
          prefetchit1 (,%rcx,8)
