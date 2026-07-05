// RUN: llvm-mc -triple x86_64-unknown-unknown %s > %t 2> %t.err
// RUN: FileCheck < %t %s
// RUN: FileCheck --check-prefix=CHECK-STDERR < %t.err %s

// CHECK: vfcmaddcph %zmm24, %zmm23, %zmm24
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph %zmm24, %zmm23, %zmm24

// CHECK: vfcmaddcph {rn-sae}, %zmm24, %zmm23, %zmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph {rn-sae}, %zmm24, %zmm23, %zmm23

// CHECK: vfcmaddcph %zmm24, %zmm23, %zmm24 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph %zmm24, %zmm23, %zmm24 {%k7}

// CHECK: vfcmaddcph {rz-sae}, %zmm24, %zmm23, %zmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph {rz-sae}, %zmm24, %zmm23, %zmm23 {%k7} {z}

// CHECK: vfcmaddcph  268435456(%rbp,%r14,8), %zmm23, %zmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph  268435456(%rbp,%r14,8), %zmm23, %zmm23

// CHECK: vfcmaddcph  291(%r8,%rax,4), %zmm23, %zmm23 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph  291(%r8,%rax,4), %zmm23, %zmm23 {%k7}

// CHECK: vfcmaddcph  (%rip){1to16}, %zmm23, %zmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph  (%rip){1to16}, %zmm23, %zmm23

// CHECK: vfcmaddcph  -2048(,%rbp,2), %zmm23, %zmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph  -2048(,%rbp,2), %zmm23, %zmm23

// CHECK: vfcmaddcph  8128(%rcx), %zmm23, %zmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph  8128(%rcx), %zmm23, %zmm23 {%k7} {z}

// CHECK: vfcmaddcph  -512(%rdx){1to16}, %zmm23, %zmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph  -512(%rdx){1to16}, %zmm23, %zmm23 {%k7} {z}

// CHECK: vfcmaddcsh %xmm24, %xmm23, %xmm24
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcsh %xmm24, %xmm23, %xmm24

// CHECK: vfcmaddcsh {rn-sae}, %xmm24, %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcsh {rn-sae}, %xmm24, %xmm23, %xmm23

// CHECK: vfcmaddcsh %xmm24, %xmm23, %xmm24 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcsh %xmm24, %xmm23, %xmm24 {%k7}

// CHECK: vfcmaddcsh {rz-sae}, %xmm24, %xmm23, %xmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcsh {rz-sae}, %xmm24, %xmm23, %xmm23 {%k7} {z}

// CHECK: vfcmaddcsh  268435456(%rbp,%r14,8), %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcsh  268435456(%rbp,%r14,8), %xmm23, %xmm23

// CHECK: vfcmaddcsh  291(%r8,%rax,4), %xmm23, %xmm23 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcsh  291(%r8,%rax,4), %xmm23, %xmm23 {%k7}

// CHECK: vfcmaddcsh  (%rip), %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcsh  (%rip), %xmm23, %xmm23

// CHECK: vfcmaddcsh  -128(,%rbp,2), %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcsh  -128(,%rbp,2), %xmm23, %xmm23

// CHECK: vfcmaddcsh  508(%rcx), %xmm23, %xmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcsh  508(%rcx), %xmm23, %xmm23 {%k7} {z}

// CHECK: vfcmaddcsh  -512(%rdx), %xmm23, %xmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcsh  -512(%rdx), %xmm23, %xmm23 {%k7} {z}

// CHECK: vfcmulcph %zmm24, %zmm23, %zmm24
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph %zmm24, %zmm23, %zmm24

// CHECK: vfcmulcph {rn-sae}, %zmm24, %zmm23, %zmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph {rn-sae}, %zmm24, %zmm23, %zmm23

// CHECK: vfcmulcph %zmm24, %zmm23, %zmm24 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph %zmm24, %zmm23, %zmm24 {%k7}

// CHECK: vfcmulcph {rz-sae}, %zmm24, %zmm23, %zmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph {rz-sae}, %zmm24, %zmm23, %zmm23 {%k7} {z}

// CHECK: vfcmulcph  268435456(%rbp,%r14,8), %zmm23, %zmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph  268435456(%rbp,%r14,8), %zmm23, %zmm23

// CHECK: vfcmulcph  291(%r8,%rax,4), %zmm23, %zmm23 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph  291(%r8,%rax,4), %zmm23, %zmm23 {%k7}

// CHECK: vfcmulcph  (%rip){1to16}, %zmm23, %zmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph  (%rip){1to16}, %zmm23, %zmm23

// CHECK: vfcmulcph  -2048(,%rbp,2), %zmm23, %zmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph  -2048(,%rbp,2), %zmm23, %zmm23

// CHECK: vfcmulcph  8128(%rcx), %zmm23, %zmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph  8128(%rcx), %zmm23, %zmm23 {%k7} {z}

// CHECK: vfcmulcph  -512(%rdx){1to16}, %zmm23, %zmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph  -512(%rdx){1to16}, %zmm23, %zmm23 {%k7} {z}

// CHECK: vfcmulcsh %xmm24, %xmm23, %xmm24
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcsh %xmm24, %xmm23, %xmm24

// CHECK: vfcmulcsh {rn-sae}, %xmm24, %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcsh {rn-sae}, %xmm24, %xmm23, %xmm23

// CHECK: vfcmulcsh %xmm24, %xmm23, %xmm24 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcsh %xmm24, %xmm23, %xmm24 {%k7}

// CHECK: vfcmulcsh {rz-sae}, %xmm24, %xmm23, %xmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcsh {rz-sae}, %xmm24, %xmm23, %xmm23 {%k7} {z}

// CHECK: vfcmulcsh  268435456(%rbp,%r14,8), %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcsh  268435456(%rbp,%r14,8), %xmm23, %xmm23

// CHECK: vfcmulcsh  291(%r8,%rax,4), %xmm23, %xmm23 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcsh  291(%r8,%rax,4), %xmm23, %xmm23 {%k7}

// CHECK: vfcmulcsh  (%rip), %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcsh  (%rip), %xmm23, %xmm23

// CHECK: vfcmulcsh  -128(,%rbp,2), %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcsh  -128(,%rbp,2), %xmm23, %xmm23

// CHECK: vfcmulcsh  508(%rcx), %xmm23, %xmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcsh  508(%rcx), %xmm23, %xmm23 {%k7} {z}

// CHECK: vfcmulcsh  -512(%rdx), %xmm23, %xmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcsh  -512(%rdx), %xmm23, %xmm23 {%k7} {z}

// CHECK: vfmaddcph %zmm24, %zmm23, %zmm24
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph %zmm24, %zmm23, %zmm24

// CHECK: vfmaddcph {rn-sae}, %zmm24, %zmm23, %zmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph {rn-sae}, %zmm24, %zmm23, %zmm23

// CHECK: vfmaddcph %zmm24, %zmm23, %zmm24 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph %zmm24, %zmm23, %zmm24 {%k7}

// CHECK: vfmaddcph {rz-sae}, %zmm24, %zmm23, %zmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph {rz-sae}, %zmm24, %zmm23, %zmm23 {%k7} {z}

// CHECK: vfmaddcph  268435456(%rbp,%r14,8), %zmm23, %zmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph  268435456(%rbp,%r14,8), %zmm23, %zmm23

// CHECK: vfmaddcph  291(%r8,%rax,4), %zmm23, %zmm23 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph  291(%r8,%rax,4), %zmm23, %zmm23 {%k7}

// CHECK: vfmaddcph  (%rip){1to16}, %zmm23, %zmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph  (%rip){1to16}, %zmm23, %zmm23

// CHECK: vfmaddcph  -2048(,%rbp,2), %zmm23, %zmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph  -2048(,%rbp,2), %zmm23, %zmm23

// CHECK: vfmaddcph  8128(%rcx), %zmm23, %zmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph  8128(%rcx), %zmm23, %zmm23 {%k7} {z}

// CHECK: vfmaddcph  -512(%rdx){1to16}, %zmm23, %zmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph  -512(%rdx){1to16}, %zmm23, %zmm23 {%k7} {z}

// CHECK: vfmaddcsh %xmm24, %xmm23, %xmm24
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcsh %xmm24, %xmm23, %xmm24

// CHECK: vfmaddcsh {rn-sae}, %xmm24, %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcsh {rn-sae}, %xmm24, %xmm23, %xmm23

// CHECK: vfmaddcsh %xmm24, %xmm23, %xmm24 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcsh %xmm24, %xmm23, %xmm24 {%k7}

// CHECK: vfmaddcsh {rz-sae}, %xmm24, %xmm23, %xmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcsh {rz-sae}, %xmm24, %xmm23, %xmm23 {%k7} {z}

// CHECK: vfmaddcsh  268435456(%rbp,%r14,8), %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcsh  268435456(%rbp,%r14,8), %xmm23, %xmm23

// CHECK: vfmaddcsh  291(%r8,%rax,4), %xmm23, %xmm23 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcsh  291(%r8,%rax,4), %xmm23, %xmm23 {%k7}

// CHECK: vfmaddcsh  (%rip), %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcsh  (%rip), %xmm23, %xmm23

// CHECK: vfmaddcsh  -128(,%rbp,2), %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcsh  -128(,%rbp,2), %xmm23, %xmm23

// CHECK: vfmaddcsh  508(%rcx), %xmm23, %xmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcsh  508(%rcx), %xmm23, %xmm23 {%k7} {z}

// CHECK: vfmaddcsh  -512(%rdx), %xmm23, %xmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcsh  -512(%rdx), %xmm23, %xmm23 {%k7} {z}

// CHECK: vfmulcph %zmm24, %zmm23, %zmm24
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph %zmm24, %zmm23, %zmm24

// CHECK: vfmulcph {rn-sae}, %zmm24, %zmm23, %zmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph {rn-sae}, %zmm24, %zmm23, %zmm23

// CHECK: vfmulcph %zmm24, %zmm23, %zmm24 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph %zmm24, %zmm23, %zmm24 {%k7}

// CHECK: vfmulcph {rz-sae}, %zmm24, %zmm23, %zmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph {rz-sae}, %zmm24, %zmm23, %zmm23 {%k7} {z}

// CHECK: vfmulcph  268435456(%rbp,%r14,8), %zmm23, %zmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph  268435456(%rbp,%r14,8), %zmm23, %zmm23

// CHECK: vfmulcph  291(%r8,%rax,4), %zmm23, %zmm23 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph  291(%r8,%rax,4), %zmm23, %zmm23 {%k7}

// CHECK: vfmulcph  (%rip){1to16}, %zmm23, %zmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph  (%rip){1to16}, %zmm23, %zmm23

// CHECK: vfmulcph  -2048(,%rbp,2), %zmm23, %zmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph  -2048(,%rbp,2), %zmm23, %zmm23

// CHECK: vfmulcph  8128(%rcx), %zmm23, %zmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph  8128(%rcx), %zmm23, %zmm23 {%k7} {z}

// CHECK: vfmulcph  -512(%rdx){1to16}, %zmm23, %zmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph  -512(%rdx){1to16}, %zmm23, %zmm23 {%k7} {z}

// CHECK: vfmulcsh %xmm24, %xmm23, %xmm24
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcsh %xmm24, %xmm23, %xmm24

// CHECK: vfmulcsh {rn-sae}, %xmm24, %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcsh {rn-sae}, %xmm24, %xmm23, %xmm23

// CHECK: vfmulcsh %xmm24, %xmm23, %xmm24 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcsh %xmm24, %xmm23, %xmm24 {%k7}

// CHECK: vfmulcsh {rz-sae}, %xmm24, %xmm23, %xmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcsh {rz-sae}, %xmm24, %xmm23, %xmm23 {%k7} {z}

// CHECK: vfmulcsh  268435456(%rbp,%r14,8), %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcsh  268435456(%rbp,%r14,8), %xmm23, %xmm23

// CHECK: vfmulcsh  291(%r8,%rax,4), %xmm23, %xmm23 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcsh  291(%r8,%rax,4), %xmm23, %xmm23 {%k7}

// CHECK: vfmulcsh  (%rip), %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcsh  (%rip), %xmm23, %xmm23

// CHECK: vfmulcsh  -128(,%rbp,2), %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcsh  -128(,%rbp,2), %xmm23, %xmm23

// CHECK: vfmulcsh  508(%rcx), %xmm23, %xmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcsh  508(%rcx), %xmm23, %xmm23 {%k7} {z}

// CHECK: vfmulcsh  -512(%rdx), %xmm23, %xmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcsh  -512(%rdx), %xmm23, %xmm23 {%k7} {z}

