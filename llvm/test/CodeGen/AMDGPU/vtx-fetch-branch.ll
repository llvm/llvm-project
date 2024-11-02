; RUN: llc -march=r600 -mcpu=redwood %s -o - | FileCheck %s

; This tests for a bug where vertex fetch clauses right before an ENDIF
; instruction where being emitted after the ENDIF.  We were using ALU_POP_AFTER
; for the ALU clause before the vetex fetch instead of emitting a POP instruction
; after the fetch clause.


; CHECK-LABEL: {{^}}test:
; CHECK-NOT: ALU_POP_AFTER
; CHECK: TEX
; CHECK-NEXT: POP
define amdgpu_kernel void @test(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %cond) {
entry:
  %0 = icmp eq i32 %cond, 0
  br i1 %0, label %endif, label %if

if:
  %1 = load i32, ptr addrspace(1) %in
  br label %endif

endif:
  %x = phi i32 [ %1, %if], [ 0, %entry]
  store i32 %x, ptr addrspace(1) %out
  br label %done

done:
  ret void
}
