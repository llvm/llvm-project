; RUN: split-file %s %t
; RUN: llc -O0 -mtriple=amdgpu9.00-amd-amdhsa \
; RUN:   < %t/ab.ll | FileCheck %s --check-prefix=AB
; RUN: llc -O0 -mtriple=amdgpu9.00-amd-amdhsa \
; RUN:   < %t/ba.ll | FileCheck %s --check-prefix=BA
; RUN: llc -O0 -mtriple=amdgpu9.00-amd-amdhsa --amdhsa-code-object-version=4 \
; RUN:   -filetype=obj %t/ab.ll -o %t/v4.o
; RUN: llvm-readobj --hex-dump=.rodata %t/v4.o | FileCheck %s --check-prefix=V4
; RUN: llc -O0 -mtriple=amdgpu9.00-amd-amdhsa --amdhsa-code-object-version=4 \
; RUN:   %t/ab.ll -o %t/v4.s
; RUN: llvm-mc -triple=amdgpu9.00-amd-amdhsa -filetype=obj %t/v4.s -o %t/v4-roundtrip.o
; RUN: llvm-readobj --hex-dump=.rodata %t/v4-roundtrip.o | FileCheck %s --check-prefix=V4
; RUN: llc -O0 -mtriple=amdgpu9.00-amd-amdhsa --amdhsa-code-object-version=5 \
; RUN:   -filetype=obj %t/ab.ll -o %t/v5.o
; RUN: llvm-readobj --hex-dump=.rodata %t/v5.o | FileCheck %s --check-prefix=V5
; RUN: llc -O0 -mtriple=amdgpu9.00-amd-amdhsa --amdhsa-code-object-version=5 \
; RUN:   %t/ab.ll -o %t/v5.s
; RUN: llvm-mc -triple=amdgpu9.00-amd-amdhsa -filetype=obj %t/v5.s -o %t/v5-roundtrip.o
; RUN: llvm-readobj --hex-dump=.rodata %t/v5-roundtrip.o | FileCheck %s --check-prefix=V5

; Both entry paths are finite and satisfy norecurse. The private-segment
; expression drops a syntactic cycle edge, so V5 and later must request
; dynamic-stack provisioning at both roots, regardless of emission order.
; AB-COUNT-2: .amdhsa_uses_dynamic_stack 1
; AB-NOT: .amdhsa_uses_dynamic_stack 0
; BA-COUNT-2: .amdhsa_uses_dynamic_stack 1
; BA-NOT: .amdhsa_uses_dynamic_stack 0

; Each kernel descriptor is 64 bytes. Its 16-bit properties field at offset 56
; contains the dynamic-stack bit (0x0800) only in V5 and later. Check the raw
; bytes because the V4 assembly printer omits the directive even if the direct
; object path incorrectly sets the bit. Both emission paths must agree.
; V4:      Hex dump of section '.rodata':
; V4:      0x00000030 {{[0-9a-f]+}} {{[0-9a-f]+}} 3f000000
; V4:      0x00000070 {{[0-9a-f]+}} {{[0-9a-f]+}} 3f000000
; V5:      Hex dump of section '.rodata':
; V5:      0x00000030 {{[0-9a-f]+}} {{[0-9a-f]+}} 3f080000
; V5:      0x00000070 {{[0-9a-f]+}} {{[0-9a-f]+}} 3f080000

;--- ab.ll
target triple = "amdgpu9.00-amd-amdhsa"

define hidden void @a(i1 %go) noinline norecurse {
entry:
  %x = alloca [32 x i8], align 4, addrspace(5)
  store volatile i8 1, ptr addrspace(5) %x
  br i1 %go, label %call, label %ret
call:
  call void @b(i1 false)
  br label %ret
ret:
  ret void
}

define hidden void @b(i1 %go) noinline norecurse {
entry:
  %x = alloca [64 x i8], align 4, addrspace(5)
  store volatile i8 1, ptr addrspace(5) %x
  br i1 %go, label %call, label %ret
call:
  call void @a(i1 false)
  br label %ret
ret:
  ret void
}

define amdgpu_kernel void @k_a() {
  call void @a(i1 true)
  ret void
}

define amdgpu_kernel void @k_b() {
  call void @b(i1 true)
  ret void
}

;--- ba.ll
target triple = "amdgpu9.00-amd-amdhsa"

define hidden void @b(i1 %go) noinline norecurse {
entry:
  %x = alloca [64 x i8], align 4, addrspace(5)
  store volatile i8 1, ptr addrspace(5) %x
  br i1 %go, label %call, label %ret
call:
  call void @a(i1 false)
  br label %ret
ret:
  ret void
}

define hidden void @a(i1 %go) noinline norecurse {
entry:
  %x = alloca [32 x i8], align 4, addrspace(5)
  store volatile i8 1, ptr addrspace(5) %x
  br i1 %go, label %call, label %ret
call:
  call void @b(i1 false)
  br label %ret
ret:
  ret void
}

define amdgpu_kernel void @k_a() {
  call void @a(i1 true)
  ret void
}

define amdgpu_kernel void @k_b() {
  call void @b(i1 true)
  ret void
}
