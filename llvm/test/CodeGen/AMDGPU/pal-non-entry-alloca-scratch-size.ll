; RUN: llc -mtriple=amdgpu12.00--amdpal < %s | FileCheck -check-prefixes=CHECK,OPT %s
; RUN: llc -mtriple=amdgpu12.00--amdpal -O0 < %s | FileCheck -check-prefixes=CHECK,O0 %s

; Reduced from an LLPC compute shader where workgroup memory initialization
; splits the entry block before the private array is allocated. The 1040 byte
; alloca (plus 16 bytes for stack realignment) must be accounted for in the
; scratch size reported to PAL.

; CHECK-LABEL: {{^}}_amdgpu_cs_main:
; CHECK: .set .L_amdgpu_cs_main.has_dyn_sized_stack, 0
; OPT: ; ScratchSize: 1056
; O0: ; ScratchSize: 1072
; CHECK: .entry_point_symbol: _amdgpu_cs_main
; OPT: .scratch_memory_size: 0x420
; O0: .scratch_memory_size: 0x430

@lds = external addrspace(3) global [64 x i32], align 4

define amdgpu_cs void @_amdgpu_cs_main(i32 %tid, i32 %idx, ptr addrspace(1) %out) {
.entry:
  %in.range = icmp ult i32 %tid, 64
  br i1 %in.range, label %.init, label %.endInit

.init:
  %lds.ptr = getelementptr [4 x i8], ptr addrspace(3) @lds, i32 %tid
  store i32 0, ptr addrspace(3) %lds.ptr, align 4
  br label %.endInit

.endInit:
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %proxy = alloca [65 x <3 x float>], align 16, addrspace(5)
  br label %loop

loop:
  %i = phi i32 [ 0, %.endInit ], [ %i.next, %loop ]
  %elt = getelementptr [16 x i8], ptr addrspace(5) %proxy, i32 %i
  %f = uitofp i32 %i to float
  %vec = insertelement <3 x float> poison, float %f, i64 0
  store <3 x float> %vec, ptr addrspace(5) %elt, align 16
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, 65
  br i1 %done, label %exit, label %loop

exit:
  %sel = urem i32 %idx, 65
  %ld.ptr = getelementptr [16 x i8], ptr addrspace(5) %proxy, i32 %sel
  %val = load i32, ptr addrspace(5) %ld.ptr, align 16
  store i32 %val, ptr addrspace(1) %out, align 4
  ret void
}
