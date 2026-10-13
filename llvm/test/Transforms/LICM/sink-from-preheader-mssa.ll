; RUN: opt -passes='loop-mssa(licm),print<memoryssa><no-ensure-optimized-uses>' -disable-output < %s 2>&1 | FileCheck %s

; The loads have different optimized clobbers, but the same unoptimized
; defining access reaches both at the beginning of the exit block.
define i32 @test_sink_different_clobbers(ptr noalias %p, ptr noalias %q, i1 %cond) {
; CHECK-LABEL: MemorySSA for function: test_sink_different_clobbers
; CHECK:       ; [[P_DEF:[0-9]+]] = MemoryDef(liveOnEntry)
; CHECK-NEXT:    store i32 1, ptr %p, align 4
; CHECK:       ; [[Q_DEF:[0-9]+]] = MemoryDef([[P_DEF]])
; CHECK-NEXT:    store i32 2, ptr %q, align 4
; CHECK:       exit:
; CHECK:       ; MemoryUse([[Q_DEF]])
; CHECK-NEXT:    %load.q = load i32, ptr %q, align 4
; CHECK:       ; MemoryUse([[Q_DEF]])
; CHECK-NEXT:    %load.p = load i32, ptr %p, align 4
;
entry:
  store i32 1, ptr %p
  store i32 2, ptr %q
  %load.q = load i32, ptr %q
  %load.p = load i32, ptr %p
  br label %loop

loop:
  br i1 %cond, label %loop, label %exit

exit:
  %add = add i32 %load.p, %load.q
  ret i32 %add
}
