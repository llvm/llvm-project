; RUN: llc < %s

; This caused ScheduleDAG to crash in EmitPhysRegCopy when searching
; the uses of a copy to a physical register without ignoring non-data
; dependence, PR10220.

define void @f(ptr nocapture %a, ptr nocapture %b, ptr nocapture %cc, ptr nocapture %dd) nounwind uwtable noinline ssp {
entry:
  %c = load i256, ptr %cc
  %d = load i256, ptr %dd
  %add = add nsw i256 %c, %d
  store i256 %add, ptr %a, align 8
  %or = or i256 %c, 1606938044258990275541962092341162602522202993782792835301376
  %add6 = add nsw i256 %or, %d
  store i256 %add6, ptr %b, align 8
  ret void
}
