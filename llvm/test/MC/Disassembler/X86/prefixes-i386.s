// RUN: llvm-mc -disassemble %s -triple=i386-apple-darwin9

// CHECK:  movl    %fs:24, %eax
0x64 0xa1 0x18 0x00 0x00 0x00 # mov eax, dword ptr fs:[18h]

# CHECK: rep
# CHECK-NEXT: insb    %dx, %es:(%rdi)
0xf3 0x6c #rep ins
# CHECK: rep
# CHECK-NEXT: insl    %dx, %es:(%rdi)
0xf3 0x6d #rep ins
# CHECK: rep
# CHECK-NEXT: movsb   (%rsi), %es:(%rdi)
0xf3 0xa4 #rep movs
# CHECK: rep
# CHECK-NEXT: movsl   (%rsi), %es:(%rdi)
0xf3 0xa5 #rep movs
# CHECK: rep
# CHECK-NEXT: outsb   (%rsi), %dx
0xf3 0x6e #rep outs
# CHECK: rep
# CHECK-NEXT: outsl   (%rsi), %dx
0xf3 0x6f #rep outs
# CHECK: rep
# CHECK-NEXT: lodsb   (%rsi), %al
0xf3 0xac #rep lods
# CHECK: rep
# CHECK-NEXT: lodsl   (%rsi), %eax
0xf3 0xad #rep lods
# CHECK: rep
# CHECK-NEXT: stosb   %al, %es:(%rdi)
0xf3 0xaa #rep stos
# CHECK: rep
# CHECK-NEXT: stosl   %eax, %es:(%rdi)
0xf3 0xab #rep stos
# CHECK: rep
# CHECK-NEXT: cmpsb   %es:(%rdi), (%rsi)
0xf3 0xa6 #rep cmps
# CHECK: rep
# CHECK-NEXT: cmpsl   %es:(%rdi), (%rsi)
0xf3 0xa7 #repe cmps
# CHECK: rep
# CHECK-NEXT: scasb   %es:(%rdi), %al
0xf3 0xae #repe scas
# CHECK: rep
# CHECK-NEXT: scasl   %es:(%rdi), %eax
0xf3 0xaf #repe scas
# CHECK: repne
# CHECK-NEXT: cmpsb   %es:(%rdi), (%rsi)
0xf2 0xa6 #repne cmps
# CHECK: repne
# CHECK-NEXT: cmpsl   %es:(%rdi), (%rsi)
0xf2 0xa7 #repne cmps
# CHECK: repne
# CHECK-NEXT: scasb   %es:(%rdi), %al
0xf2 0xae #repne scas
# CHECK: repne
# CHECK-NEXT: scasl   %es:(%rdi), %eax
0xf2 0xaf #repne scas

// CHECK: mulsd   %xmm7, %xmm7
0x66 0xF3 0xF2 0x0F 0x59 0xFF
// CHECK: mulss   %xmm7, %xmm7
0x66 0xF2 0xF3 0x0F 0x59 0xFF
// CHECK: mulpd   %xmm7, %xmm7
0x66 0x0F 0x59 0xFF
// CHECK: mulsd   %xmm7, %xmm7
0xf2 0x66 0x0f 0x59 0xff
