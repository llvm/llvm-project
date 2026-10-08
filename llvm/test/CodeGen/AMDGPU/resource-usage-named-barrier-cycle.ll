; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa < %s | FileCheck %s

; a and b call each other, and only a uses a named barrier. kb needs that
; one barrier, not the module maximum that d's 16 barriers set. The
; descriptor counts barriers in blocks of four.

; CHECK-LABEL: .amdhsa_kernel kb
; CHECK: .amdhsa_named_barrier_count 1
; CHECK-LABEL: .amdhsa_kernel kd
; CHECK: .amdhsa_named_barrier_count 4

@bar = internal addrspace(15) global target("amdgcn.named.barrier", 0) poison
@bar16 = internal addrspace(15) global [16 x target("amdgcn.named.barrier", 0)] poison

define void @a(i1 %go) {
  call void @llvm.amdgcn.s.barrier.join(ptr addrspace(15) @bar)
  br i1 %go, label %call, label %done
call:
  call void @b(i1 false)
  br label %done
done:
  ret void
}

define void @b(i1 %go) {
  br i1 %go, label %call, label %done
call:
  call void @a(i1 false)
  br label %done
done:
  ret void
}

define void @d() {
  call void @llvm.amdgcn.s.barrier.join(ptr addrspace(15) @bar16)
  ret void
}

define amdgpu_kernel void @kb() {
  call void @b(i1 true)
  ret void
}

define amdgpu_kernel void @kd() {
  call void @d()
  ret void
}
