; NOTE: Do not autogenerate. Checks relate spill-slot identities across
; full-width and partial spill modes.
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -stress-regalloc=2 -stop-after=greedy -verify-machineinstrs -verify-regalloc < %s | FileCheck %s --check-prefixes=CHECK,FULL --implicit-check-not=SI_SPILL_S32_SAVE_partial
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -stress-regalloc=2 -stop-after=greedy -enable-partial-spills -verify-machineinstrs -verify-regalloc < %s | FileCheck %s --check-prefixes=CHECK,PARTIAL
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -stress-regalloc=2 -enable-partial-spills -verify-machineinstrs -verify-regalloc -filetype=null < %s

; Like a wide load of scalar arguments, each tuple has words used early and a
; word needed throughout a later loop. Volatile accesses keep those uses ordered.
; Limit registers so greedy creates the partial split from ordinary LLVM IR.
; Only word 2 of the second tuple needs saving after its early uses.

; CHECK-LABEL: name: spill_loaded_tuple_word
; CHECK: body:
; CHECK: [[ARGS:%[0-9]+]]:sgpr_128 = S_LOAD_DWORDX4_IMM {{.*}}, 16, 0 :: (volatile load
; FULL-NEXT: SI_SPILL_S128_SAVE [[ARGS]], [[SLOT:%stack\.[0-9]+]],
; CHECK: S_MUL_I32 [[ARGS]].sub0, [[ARGS]].sub1
; PARTIAL-NEXT: [[PART:%[0-9]+]].sub2:sgpr_128 = lr-split COPY [[ARGS]].sub2
; PARTIAL-NEXT: SI_SPILL_S32_SAVE_partial [[PART]].sub2, [[SLOT:%stack\.[0-9]+]], 8,
; PARTIAL-NOT: SI_SPILL_S128_SAVE {{.*}}, [[SLOT]],
; CHECK: [[RELOAD:%[0-9]+]]:sgpr_128 = SI_SPILL_S128_RESTORE [[SLOT]],
; CHECK-NEXT: [[LATE:%[0-9]+]].sub2:sgpr_128 = lr-split COPY [[RELOAD]].sub2
; CHECK: S_XOR_B32 {{%[0-9]+}}, [[LATE]].sub2,
; CHECK: S_ENDPGM

define amdgpu_kernel void @spill_loaded_tuple_word(ptr addrspace(1) %out, ptr addrspace(4) %input, i32 %count) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %out.ptr = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  %first = load volatile <4 x i32>, ptr addrspace(4) %input, align 16
  %first.a = extractelement <4 x i32> %first, i32 0
  %first.b = extractelement <4 x i32> %first, i32 1
  %first.late = extractelement <4 x i32> %first, i32 2
  %first.product = mul i32 %first.a, %first.b
  store volatile i32 %first.product, ptr addrspace(1) %out.ptr
  %second.ptr = getelementptr <4 x i32>, ptr addrspace(4) %input, i32 1
  %second = load volatile <4 x i32>, ptr addrspace(4) %second.ptr, align 16
  %second.a = extractelement <4 x i32> %second, i32 0
  %second.b = extractelement <4 x i32> %second, i32 1
  %second.late = extractelement <4 x i32> %second, i32 2
  %second.product = mul i32 %second.a, %second.b
  store volatile i32 %second.product, ptr addrspace(1) %out.ptr
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %next, %loop ]
  %product = mul i32 %sum, %first.late
  %next = xor i32 %product, %second.late
  store volatile i32 %next, ptr addrspace(1) %out.ptr
  %inc = add i32 %i, 1
  %done = icmp eq i32 %inc, %count
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
