; NOTE: Do not autogenerate. Checks relate spill-slot identities across
; full-width and partial spill modes.
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -stress-regalloc=2 -stop-after=greedy -verify-machineinstrs -verify-regalloc < %s | FileCheck %s --check-prefixes=CHECK,FULL --implicit-check-not=SI_SPILL_S32_SAVE_partial
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -stress-regalloc=2 -stop-after=greedy -enable-partial-spills -verify-machineinstrs -verify-regalloc < %s | FileCheck %s --check-prefixes=CHECK,PARTIAL
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -stress-regalloc=2 -enable-partial-spills -verify-machineinstrs -verify-regalloc -filetype=null < %s

; A buffer resource requires four SGPR words, but its fields can be defined at
; different times. Greedy splits the partially constructed resource under
; pressure. Save each available word at its original offset, then complete the
; resource before the buffer load. A fully constructed resource still spills
; at full width. The LLVM IR supplies every field and no split metadata.

; CHECK-LABEL: name: spill_partial_buffer_resource
; CHECK: body:
; CHECK: [[FIELDS:%[0-9]+]].sub3:sgpr_128 = S_MOV_B32 131072
; CHECK-NEXT: [[FIELDS]].sub2:sgpr_128 = S_MOV_B32 1024
; FULL-NEXT: SI_SPILL_S128_SAVE [[FIELDS]], [[SLOT:%stack\.[0-9]+]],
; PARTIAL-NEXT: [[PART:%[0-9]+]].sub2_sub3:sgpr_128 = lr-split COPY [[FIELDS]].sub2_sub3
; PARTIAL-NEXT: SI_SPILL_S32_SAVE_partial [[PART]].sub2, [[SLOT:%stack\.[0-9]+]], 8,
; PARTIAL-NEXT: SI_SPILL_S32_SAVE_partial [[PART]].sub3, [[SLOT]], 12,
; CHECK: [[RELOAD:%[0-9]+]]:sgpr_128 = SI_SPILL_S128_RESTORE [[SLOT]],
; CHECK-NEXT: [[UPPER:%[0-9]+]].sub2_sub3:sgpr_128 = lr-split COPY [[RELOAD]].sub2_sub3
; CHECK-NEXT: [[UPPER]].sub1:sgpr_128 = S_AND_B32
; CHECK-NEXT: [[PART2:%[0-9]+]].sub2_sub3:sgpr_128 = lr-split COPY [[UPPER]].sub2_sub3 {
; CHECK-NEXT: internal [[PART2]].sub1:sgpr_128 = lr-split COPY [[UPPER]].sub1
; CHECK-NEXT: }
; FULL-NEXT: SI_SPILL_S128_SAVE [[PART2]], [[SLOT]],
; PARTIAL-NEXT: SI_SPILL_S32_SAVE_partial [[PART2]].sub1, [[SLOT]], 4,
; PARTIAL-NEXT: SI_SPILL_S32_SAVE_partial [[PART2]].sub2, [[SLOT]], 8,
; PARTIAL-NEXT: SI_SPILL_S32_SAVE_partial [[PART2]].sub3, [[SLOT]], 12,
; CHECK: [[COMPLETE:%[0-9]+]]:sgpr_128 = SI_SPILL_S128_RESTORE [[SLOT]],
; CHECK-NEXT: [[COMPLETE]].sub0:sgpr_128 = COPY
; CHECK-NEXT: SI_SPILL_S128_SAVE [[COMPLETE]], [[SLOT]],
; CHECK-NEXT: [[RESOURCE:%[0-9]+]]:sgpr_128 = SI_SPILL_S128_RESTORE [[SLOT]],
; CHECK: BUFFER_LOAD_DWORD_OFFEN {{%[0-9]+}}, [[RESOURCE]],
; CHECK: S_ENDPGM

define amdgpu_kernel void @spill_partial_buffer_resource(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %count) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %offset = shl i32 %tid, 2
  %resource = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1.i64(ptr addrspace(1) %in, i16 0, i64 1024, i32 131072)
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %next, %loop ]
  %value = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %resource, i32 %offset, i32 0, i32 0)
  %next = add i32 %sum, %value
  store atomic i32 %next, ptr addrspace(1) %out monotonic, align 4
  %inc = add i32 %i, 1
  %done = icmp eq i32 %inc, %count
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
