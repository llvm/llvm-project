; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx1200 -filetype=null %s 2>&1 | FileCheck --check-prefix=WARN %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx1200 -amdgpu-assume-dynamic-stack-object-size=1024 -filetype=null %s 2>&1 | FileCheck --allow-empty --check-prefix=NOWARN %s
; RUN: llc -mtriple=amdgcn--amdmesa3d -mcpu=gfx1200 -filetype=null %s 2>&1 | FileCheck --allow-empty --check-prefix=NOWARN %s

; WARN: warning: {{.*}} in function loop_alloca void (i32): scratch size includes only an assumed size for dynamic stack allocations
; WARN-NOT: dynamic stack allocations
; NOWARN-NOT: dynamic stack allocations

define amdgpu_cs void @loop_alloca(i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %a = alloca [16 x i32], align 4, addrspace(5)
  %gep = getelementptr [16 x i32], ptr addrspace(5) %a, i32 0, i32 %i
  store volatile i32 %i, ptr addrspace(5) %gep
  %i.next = add i32 %i, 1
  %cmp = icmp ult i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define amdgpu_cs void @entry_alloca(i32 %i) {
  %a = alloca [16 x i32], align 4, addrspace(5)
  %gep = getelementptr [16 x i32], ptr addrspace(5) %a, i32 0, i32 %i
  store volatile i32 %i, ptr addrspace(5) %gep
  ret void
}
