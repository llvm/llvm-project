; RUN: opt -aa-pipeline=basic-aa,tbaa -passes='print<memoryssa><no-ensure-optimized-uses>' -disable-output < %s 2>&1 | FileCheck %s

define i8 @test1_yes(ptr %a, ptr %b) {
; CHECK: 1 = MemoryDef(liveOnEntry)
  store i8 0, ptr %a, align 1
; CHECK: MemoryUse(liveOnEntry)
  %y = load i8, ptr %b, align 1, !tbaa !0
  ret i8 %y
}

!0 = !{!1, !1, i64 0, i1 true}
!1 = !{!"qux", !2}
!2 = !{}
