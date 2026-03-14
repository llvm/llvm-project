// RUN: llvm-mc -triple x86_64-unknown-unknown %s > %t 2> %t.err
// RUN: FileCheck < %t %s
// RUN: FileCheck --check-prefix=CHECK-STDERR < %t.err %s

// CHECK: vfcmaddcph %ymm24, %ymm23, %ymm24
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph %ymm24, %ymm23, %ymm24

// CHECK: vfcmaddcph %ymm24, %ymm23, %ymm23 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph %ymm24, %ymm23, %ymm23 {%k7}

// CHECK: vfcmaddcph %ymm24, %ymm23, %ymm24 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph %ymm24, %ymm23, %ymm24 {%k7} {z}

// CHECK: vfcmaddcph %xmm24, %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph %xmm24, %xmm23, %xmm23

// CHECK: vfcmaddcph %xmm24, %xmm23, %xmm24 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph %xmm24, %xmm23, %xmm24 {%k7}

// CHECK: vfcmaddcph %xmm24, %xmm23, %xmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph %xmm24, %xmm23, %xmm23 {%k7} {z}

// CHECK: vfcmaddcph  268435456(%rbp,%r14,8), %ymm23, %ymm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph  268435456(%rbp,%r14,8), %ymm23, %ymm23

// CHECK: vfcmaddcph  291(%r8,%rax,4), %ymm23, %ymm23 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph  291(%r8,%rax,4), %ymm23, %ymm23 {%k7}

// CHECK: vfcmaddcph  (%rip){1to8}, %ymm23, %ymm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph  (%rip){1to8}, %ymm23, %ymm23

// CHECK: vfcmaddcph  -1024(,%rbp,2), %ymm23, %ymm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph  -1024(,%rbp,2), %ymm23, %ymm23

// CHECK: vfcmaddcph  4064(%rcx), %ymm23, %ymm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph  4064(%rcx), %ymm23, %ymm23 {%k7} {z}

// CHECK: vfcmaddcph  -512(%rdx){1to8}, %ymm23, %ymm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph  -512(%rdx){1to8}, %ymm23, %ymm23 {%k7} {z}

// CHECK: vfcmaddcph  268435456(%rbp,%r14,8), %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph  268435456(%rbp,%r14,8), %xmm23, %xmm23

// CHECK: vfcmaddcph  291(%r8,%rax,4), %xmm23, %xmm23 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph  291(%r8,%rax,4), %xmm23, %xmm23 {%k7}

// CHECK: vfcmaddcph  (%rip){1to4}, %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph  (%rip){1to4}, %xmm23, %xmm23

// CHECK: vfcmaddcph  -512(,%rbp,2), %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph  -512(,%rbp,2), %xmm23, %xmm23

// CHECK: vfcmaddcph  2032(%rcx), %xmm23, %xmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph  2032(%rcx), %xmm23, %xmm23 {%k7} {z}

// CHECK: vfcmaddcph  -512(%rdx){1to4}, %xmm23, %xmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmaddcph  -512(%rdx){1to4}, %xmm23, %xmm23 {%k7} {z}

// CHECK: vfcmulcph %ymm24, %ymm23, %ymm24
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph %ymm24, %ymm23, %ymm24

// CHECK: vfcmulcph %ymm24, %ymm23, %ymm23 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph %ymm24, %ymm23, %ymm23 {%k7}

// CHECK: vfcmulcph %ymm24, %ymm23, %ymm24 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph %ymm24, %ymm23, %ymm24 {%k7} {z}

// CHECK: vfcmulcph %xmm24, %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph %xmm24, %xmm23, %xmm23

// CHECK: vfcmulcph %xmm24, %xmm23, %xmm24 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph %xmm24, %xmm23, %xmm24 {%k7}

// CHECK: vfcmulcph %xmm24, %xmm23, %xmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph %xmm24, %xmm23, %xmm23 {%k7} {z}

// CHECK: vfcmulcph  268435456(%rbp,%r14,8), %ymm23, %ymm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph  268435456(%rbp,%r14,8), %ymm23, %ymm23

// CHECK: vfcmulcph  291(%r8,%rax,4), %ymm23, %ymm23 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph  291(%r8,%rax,4), %ymm23, %ymm23 {%k7}

// CHECK: vfcmulcph  (%rip){1to8}, %ymm23, %ymm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph  (%rip){1to8}, %ymm23, %ymm23

// CHECK: vfcmulcph  -1024(,%rbp,2), %ymm23, %ymm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph  -1024(,%rbp,2), %ymm23, %ymm23

// CHECK: vfcmulcph  4064(%rcx), %ymm23, %ymm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph  4064(%rcx), %ymm23, %ymm23 {%k7} {z}

// CHECK: vfcmulcph  -512(%rdx){1to8}, %ymm23, %ymm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph  -512(%rdx){1to8}, %ymm23, %ymm23 {%k7} {z}

// CHECK: vfcmulcph  268435456(%rbp,%r14,8), %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph  268435456(%rbp,%r14,8), %xmm23, %xmm23

// CHECK: vfcmulcph  291(%r8,%rax,4), %xmm23, %xmm23 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph  291(%r8,%rax,4), %xmm23, %xmm23 {%k7}

// CHECK: vfcmulcph  (%rip){1to4}, %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph  (%rip){1to4}, %xmm23, %xmm23

// CHECK: vfcmulcph  -512(,%rbp,2), %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph  -512(,%rbp,2), %xmm23, %xmm23

// CHECK: vfcmulcph  2032(%rcx), %xmm23, %xmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph  2032(%rcx), %xmm23, %xmm23 {%k7} {z}

// CHECK: vfcmulcph  -512(%rdx){1to4}, %xmm23, %xmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfcmulcph  -512(%rdx){1to4}, %xmm23, %xmm23 {%k7} {z}

// CHECK: vfmaddcph %ymm24, %ymm23, %ymm24
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph %ymm24, %ymm23, %ymm24

// CHECK: vfmaddcph %ymm24, %ymm23, %ymm23 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph %ymm24, %ymm23, %ymm23 {%k7}

// CHECK: vfmaddcph %ymm24, %ymm23, %ymm24 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph %ymm24, %ymm23, %ymm24 {%k7} {z}

// CHECK: vfmaddcph %xmm24, %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph %xmm24, %xmm23, %xmm23

// CHECK: vfmaddcph %xmm24, %xmm23, %xmm24 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph %xmm24, %xmm23, %xmm24 {%k7}

// CHECK: vfmaddcph %xmm24, %xmm23, %xmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph %xmm24, %xmm23, %xmm23 {%k7} {z}

// CHECK: vfmaddcph  268435456(%rbp,%r14,8), %ymm23, %ymm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph  268435456(%rbp,%r14,8), %ymm23, %ymm23

// CHECK: vfmaddcph  291(%r8,%rax,4), %ymm23, %ymm23 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph  291(%r8,%rax,4), %ymm23, %ymm23 {%k7}

// CHECK: vfmaddcph  (%rip){1to8}, %ymm23, %ymm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph  (%rip){1to8}, %ymm23, %ymm23

// CHECK: vfmaddcph  -1024(,%rbp,2), %ymm23, %ymm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph  -1024(,%rbp,2), %ymm23, %ymm23

// CHECK: vfmaddcph  4064(%rcx), %ymm23, %ymm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph  4064(%rcx), %ymm23, %ymm23 {%k7} {z}

// CHECK: vfmaddcph  -512(%rdx){1to8}, %ymm23, %ymm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph  -512(%rdx){1to8}, %ymm23, %ymm23 {%k7} {z}

// CHECK: vfmaddcph  268435456(%rbp,%r14,8), %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph  268435456(%rbp,%r14,8), %xmm23, %xmm23

// CHECK: vfmaddcph  291(%r8,%rax,4), %xmm23, %xmm23 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph  291(%r8,%rax,4), %xmm23, %xmm23 {%k7}

// CHECK: vfmaddcph  (%rip){1to4}, %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph  (%rip){1to4}, %xmm23, %xmm23

// CHECK: vfmaddcph  -512(,%rbp,2), %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph  -512(,%rbp,2), %xmm23, %xmm23

// CHECK: vfmaddcph  2032(%rcx), %xmm23, %xmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph  2032(%rcx), %xmm23, %xmm23 {%k7} {z}

// CHECK: vfmaddcph  -512(%rdx){1to4}, %xmm23, %xmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmaddcph  -512(%rdx){1to4}, %xmm23, %xmm23 {%k7} {z}

// CHECK: vfmulcph %ymm24, %ymm23, %ymm24
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph %ymm24, %ymm23, %ymm24

// CHECK: vfmulcph %ymm24, %ymm23, %ymm23 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph %ymm24, %ymm23, %ymm23 {%k7}

// CHECK: vfmulcph %ymm24, %ymm23, %ymm24 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph %ymm24, %ymm23, %ymm24 {%k7} {z}

// CHECK: vfmulcph %xmm24, %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph %xmm24, %xmm23, %xmm23

// CHECK: vfmulcph %xmm24, %xmm23, %xmm24 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph %xmm24, %xmm23, %xmm24 {%k7}

// CHECK: vfmulcph %xmm24, %xmm23, %xmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph %xmm24, %xmm23, %xmm23 {%k7} {z}

// CHECK: vfmulcph  268435456(%rbp,%r14,8), %ymm23, %ymm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph  268435456(%rbp,%r14,8), %ymm23, %ymm23

// CHECK: vfmulcph  291(%r8,%rax,4), %ymm23, %ymm23 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph  291(%r8,%rax,4), %ymm23, %ymm23 {%k7}

// CHECK: vfmulcph  (%rip){1to8}, %ymm23, %ymm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph  (%rip){1to8}, %ymm23, %ymm23

// CHECK: vfmulcph  -1024(,%rbp,2), %ymm23, %ymm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph  -1024(,%rbp,2), %ymm23, %ymm23

// CHECK: vfmulcph  4064(%rcx), %ymm23, %ymm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph  4064(%rcx), %ymm23, %ymm23 {%k7} {z}

// CHECK: vfmulcph  -512(%rdx){1to8}, %ymm23, %ymm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph  -512(%rdx){1to8}, %ymm23, %ymm23 {%k7} {z}

// CHECK: vfmulcph  268435456(%rbp,%r14,8), %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph  268435456(%rbp,%r14,8), %xmm23, %xmm23

// CHECK: vfmulcph  291(%r8,%rax,4), %xmm23, %xmm23 {%k7}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph  291(%r8,%rax,4), %xmm23, %xmm23 {%k7}

// CHECK: vfmulcph  (%rip){1to4}, %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph  (%rip){1to4}, %xmm23, %xmm23

// CHECK: vfmulcph  -512(,%rbp,2), %xmm23, %xmm23
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph  -512(,%rbp,2), %xmm23, %xmm23

// CHECK: vfmulcph  2032(%rcx), %xmm23, %xmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph  2032(%rcx), %xmm23, %xmm23 {%k7} {z}

// CHECK: vfmulcph  -512(%rdx){1to4}, %xmm23, %xmm23 {%k7} {z}
// CHECK-STDERR: warning: Destination register should be distinct from source registers
          vfmulcph  -512(%rdx){1to4}, %xmm23, %xmm23 {%k7} {z}

