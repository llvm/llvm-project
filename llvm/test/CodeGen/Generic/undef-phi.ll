; RUN: llc < %s -verify-machineinstrs -verify-coalescing
;
; This function has a PHI with one undefined input. Verify that PHIElimination
; inserts an IMPLICIT_DEF instruction in the predecessor so all paths to the use
; pass through a def.

%struct.xx_stack = type { i32, ptr }

define i32 @push(ptr %stack) nounwind uwtable readonly ssp {
entry:
  %tobool1 = icmp eq ptr %stack, null
  br i1 %tobool1, label %for.end, label %for.body

for.body:
  %stack.addr.02 = phi ptr [ %0, %for.body ], [ %stack, %entry ]
  %next = getelementptr inbounds %struct.xx_stack, ptr %stack.addr.02, i64 0, i32 1
  %0 = load ptr, ptr %next, align 8
  %tobool = icmp eq ptr %0, null
  br i1 %tobool, label %for.end, label %for.body

for.end:
  %top.0.lcssa = phi ptr [ undef, %entry ], [ %stack.addr.02, %for.body ]
  %1 = load i32, ptr %top.0.lcssa, align 4
  ret i32 %1
}
