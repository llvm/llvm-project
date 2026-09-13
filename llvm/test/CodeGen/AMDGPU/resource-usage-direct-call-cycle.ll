; RUN: split-file %s %t
; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa %t/input.ll -o %t/sdag.s
; RUN: FileCheck %s < %t/sdag.s
; RUN: cat %t/sdag.s %t/check.s | llvm-mc -triple=amdgpu9.00-amd-amdhsa \
; RUN:   -filetype=obj | llvm-readobj --hex-dump=.resource_check - | \
; RUN:   FileCheck %s --check-prefix=RESOURCES
; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa -global-isel=1 \
; RUN:   -global-isel-abort=1 %t/input.ll -o %t/gisel.s
; RUN: FileCheck %s < %t/gisel.s
; RUN: cat %t/gisel.s %t/check.s | llvm-mc -triple=amdgpu9.00-amd-amdhsa \
; RUN:   -filetype=obj | llvm-readobj --hex-dump=.resource_check - | \
; RUN:   FileCheck %s --check-prefix=RESOURCES
;
; The syntactic cycle terminates on every path: a(0) calls b(1), which
; returns; b(0) calls a(1), which calls c(1). All norecurse attributes are
; therefore valid. The selected emission order closes the cycle at b, and b
; must retain c's resources through a even though it does not call c directly.
; The outgoing c edge contributes v47 and a dynamically sized frame.
;
; CHECK: .set .Lc.num_vgpr, 48
; CHECK: .amdhsa_kernel kernel_b
; CHECK: .amdhsa_uses_dynamic_stack 1
; CHECK: .amdhsa_next_free_vgpr 48
; CHECK: .name: kernel_b
; CHECK: .uses_dynamic_stack: true
; CHECK: .vgpr_count: 48
;
; The first row is the final VGPR count of a, b, c, and a padding word.
; The second row is their dynamic-stack flag. Check its own closure rather
; than relying on the separately propagated recursion flag to hide a loss.
; RESOURCES:      Hex dump of section '.resource_check':
; RESOURCES-NEXT: 0x00000000 30000000 30000000 30000000 00000000
; RESOURCES-NEXT: 0x00000010 01000000 01000000 01000000 00000000

;--- input.ll
target triple = "amdgpu9.00-amd-amdhsa"

define hidden void @b(i32 %mode) noinline optnone norecurse {
  %is0 = icmp eq i32 %mode, 0
  br i1 %is0, label %do_call, label %done
do_call:
  call void @a(i32 1)
  br label %done
done:
  ret void
}

define hidden void @a(i32 %mode) noinline optnone norecurse {
  %is0 = icmp eq i32 %mode, 0
  br i1 %is0, label %call_b, label %call_c
call_b:
  call void @b(i32 1)
  br label %done
call_c:
  call void @c(i32 1)
  br label %done
done:
  ret void
}

define hidden void @c(i32 %mode) noinline optnone norecurse {
  %frame = alloca i8, i32 %mode, align 16, addrspace(5)
  store volatile i8 1, ptr addrspace(5) %frame, align 16
  call void asm sideeffect "v_mov_b32 v47, 0", "~{v47}"()
  ret void
}

define amdgpu_kernel void @kernel_b() {
  call void @b(i32 0)
  ret void
}

;--- check.s
.section .resource_check, "", @progbits
.long .La.num_vgpr, .Lb.num_vgpr, .Lc.num_vgpr, 0
.long .La.has_dyn_sized_stack, .Lb.has_dyn_sized_stack, .Lc.has_dyn_sized_stack, 0
