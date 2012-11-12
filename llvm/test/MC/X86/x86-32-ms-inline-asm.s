// RUN: llvm-mc -x86-asm-syntax=intel -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

mov eax, [ebx].0
mov [ebx].4, ecx

// CHECK: movl (%ebx), %eax
// CHECK: encoding: [0x8b,0x03]
// CHECK: movl %ecx, 4(%ebx)
// CHECK: encoding: [0x89,0x4b,0x04]
        
_t21:                                   ## @t21
// CHECK: t21
	mov eax, [4*eax + 4]
// CHECK: movl 4(,%eax,4), %eax
// CHECK: # encoding: [0x8b,0x04,0x85,0x04,0x00,0x00,0x00]
    mov eax, [4*eax][4]
// CHECK: movl 4(,%eax,4), %eax
// CHECK: # encoding: [0x8b,0x04,0x85,0x04,0x00,0x00,0x00]
        
	mov eax, [esi + eax]
// CHECK: movl (%esi,%eax), %eax
// CHECK: # encoding: [0x8b,0x04,0x06]
	mov eax, [esi][eax]
// CHECK: movl (%esi,%eax), %eax
// CHECK: # encoding: [0x8b,0x04,0x06]
        
	mov eax, [esi + 4*eax]
// CHECK: movl (%esi,%eax,4), %eax
// CHECK: # encoding: [0x8b,0x04,0x86]
	mov eax, [esi][4*eax]
// CHECK: movl (%esi,%eax,4), %eax
// CHECK: # encoding: [0x8b,0x04,0x86]

    mov eax, [esi + eax + 4]
// CHECK: movl 4(%esi,%eax), %eax
// CHECK: # encoding: [0x8b,0x44,0x06,0x04]
	mov eax, [esi][eax + 4]
// CHECK: movl 4(%esi,%eax), %eax
// CHECK: # encoding: [0x8b,0x44,0x06,0x04]
	mov eax, [esi + eax][4]
// CHECK: movl 4(%esi,%eax), %eax
// CHECK: # encoding: [0x8b,0x44,0x06,0x04]
	mov eax, [esi][eax][4]
// CHECK: movl 4(%esi,%eax), %eax
// CHECK: # encoding: [0x8b,0x44,0x06,0x04]

	mov eax, [esi + 2*eax + 4]
// CHECK: movl 4(%esi,%eax,2), %eax
// CHECK: # encoding: [0x8b,0x44,0x46,0x04]
	mov eax, [esi][2*eax + 4]
// CHECK: movl 4(%esi,%eax,2), %eax
// CHECK: # encoding: [0x8b,0x44,0x46,0x04]
	mov eax, [esi + 2*eax][4]
// CHECK: movl 4(%esi,%eax,2), %eax
// CHECK: # encoding: [0x8b,0x44,0x46,0x04]
	mov eax, [esi][2*eax][4]
// CHECK: movl 4(%esi,%eax,2), %eax
// CHECK: # encoding: [0x8b,0x44,0x46,0x04]

	ret
