; RUN: llc < %s -verify-machineinstrs -verify-coalescing
;
; This function has a PHI with one undefined input. Verify that PHIElimination
; inserts an IMPLICIT_DEF instruction in the predecessor so all paths to the use
; pass through a def.

%struct.xx_stack = type { i32, %struct.xx_stack* }

define i32 @push(%struct.xx_stack* %stack) nounwind uwtable readonly ssp {
entry:
  %tobool1 = icmp eq %struct.xx_stack* %stack, null
  br i1 %tobool1, label %for.end, label %for.body

for.body:
  %stack.addr.02 = phi %struct.xx_stack* [ %0, %for.body ], [ %stack, %entry ]
  %next = getelementptr inbounds %struct.xx_stack* %stack.addr.02, i64 0, i32 1
  %0 = load %struct.xx_stack** %next, align 8
  %tobool = icmp eq %struct.xx_stack* %0, null
  br i1 %tobool, label %for.end, label %for.body

for.end:
  %top.0.lcssa = phi %struct.xx_stack* [ undef, %entry ], [ %stack.addr.02, %for.body ]
  %first = getelementptr inbounds %struct.xx_stack* %top.0.lcssa, i64 0, i32 0
  %1 = load i32* %first, align 4
  ret i32 %1
}
