; RUN: llc -march=r600 -mcpu=redwood -verify-machineinstrs < %s | FileCheck %s
;
; This test checks that the lds input queue will is empty at the end of
; the ALU clause.

; CHECK-LABEL: {{^}}lds_input_queue:
; CHECK: LDS_READ_RET * OQAP
; CHECK-NOT: ALU clause
; CHECK: MOV * T{{[0-9]\.[XYZW]}}, OQAP

@local_mem = internal unnamed_addr addrspace(3) global [2 x i32] undef, align 4

define void @lds_input_queue(i32 addrspace(1)* %out, i32 addrspace(1)* %in, i32 %index) {
entry:
  %0 = getelementptr inbounds [2 x i32], [2 x i32] addrspace(3)* @local_mem, i32 0, i32 %index
  %1 = load i32, i32 addrspace(3)* %0
  call void @llvm.r600.group.barrier()

  ; This will start a new clause for the vertex fetch
  %2 = load i32, i32 addrspace(1)* %in
  %3 = add i32 %1, %2
  store i32 %3, i32 addrspace(1)* %out
  ret void
}

declare void @llvm.r600.group.barrier() nounwind convergent

; The machine scheduler does not do proper alias analysis and assumes that
; loads from global values (Note that a global value is different that a
; value from global memory.  A global value is a value that is declared
; outside of a function, it can reside in any address space) alias with
; all other loads.
;
; This is a problem for scheduling the reads from the local data share (lds).
; These reads are implemented using two instructions.  The first copies the
; data from lds into the lds output queue, and the second moves the data from
; the input queue into main memory.  These two instructions don't have to be
; scheduled one after the other, but they do need to be scheduled in the same
; clause.  The aliasing problem mentioned above causes problems when there is a
; load from global memory which immediately follows a load from a global value that
; has been declared in the local memory space:
;
;  %0 = getelementptr inbounds [2 x i32], [2 x i32] addrspace(3)* @local_mem, i32 0, i32 %index
;  %1 = load i32, i32 addrspace(3)* %0
;  %2 = load i32, i32 addrspace(1)* %in
;
; The instruction selection phase will generate ISA that looks like this:
; %OQAP = LDS_READ_RET
; %vreg0 = MOV %OQAP
; %vreg1 = VTX_READ_32
; %vreg2 = ADD_INT %vreg1, %vreg0
;
; The bottom scheduler will schedule the two ALU instructions first:
;
; UNSCHEDULED:
; %OQAP = LDS_READ_RET
; %vreg1 = VTX_READ_32
;
; SCHEDULED:
;
; vreg0 = MOV %OQAP
; vreg2 = ADD_INT %vreg1, %vreg2
;
; The lack of proper aliasing results in the local memory read (LDS_READ_RET)
; to consider the global memory read (VTX_READ_32) has a chain dependency, so
; the global memory read will always be scheduled first.  This will give us a
; final program which looks like this:
;
; Alu clause:
; %OQAP = LDS_READ_RET
; VTX clause:
; %vreg1 = VTX_READ_32
; Alu clause:
; vreg0 = MOV %OQAP
; vreg2 = ADD_INT %vreg1, %vreg2
;
; This is an illegal program because the OQAP def and use know occur in
; different ALU clauses.
;
; This test checks this scenario and makes sure it doesn't result in an
; illegal program.  For now, we have fixed this issue by merging the
; LDS_READ_RET and MOV together during instruction selection and then
; expanding them after scheduling.  Once the scheduler has better alias
; analysis, we should be able to keep these instructions sparate before
; scheduling.
;
; CHECK-LABEL: {{^}}local_global_alias:
; CHECK: LDS_READ_RET
; CHECK-NOT: ALU clause
; CHECK: MOV * T{{[0-9]\.[XYZW]}}, OQAP
define void @local_global_alias(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
entry:
  %0 = getelementptr inbounds [2 x i32], [2 x i32] addrspace(3)* @local_mem, i32 0, i32 0
  %1 = load i32, i32 addrspace(3)* %0
  %2 = load i32, i32 addrspace(1)* %in
  %3 = add i32 %2, %1
  store i32 %3, i32 addrspace(1)* %out
  ret void
}
