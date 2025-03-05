; RUN: opt -S -passes=verify < %s

; This file contains MMRA metadata that is okay and should pass the verifier.

define void @test(ptr %ptr) {
  %ld = load i8, ptr %ptr,                                !mmra !0
  store i8 1, ptr %ptr,                                   !mmra !1
  call void @writesMem(),                                 !mmra !2
  call void @readsMem(),                                  !mmra !2
  fence release,                                          !mmra !0
  %rmw.1 = atomicrmw add ptr %ptr, i8 0 release,          !mmra !0
  %rmw.2 = atomicrmw add ptr %ptr, i8 0 acquire,          !mmra !0
  %pair = cmpxchg ptr %ptr, i8 0, i8 1 acquire acquire,   !mmra !1
  %ld.atomic = load atomic i8, ptr %ptr acquire, align 4, !mmra !1
  store atomic i8 1, ptr %ptr release, align 4,           !mmra !2
  %mld = call <2 x i64> @llvm.vp.load.v2i64.p0(ptr undef, <2 x i1> undef, i32 undef), !mmra !2
  ; TODO: barrier
  ret void
}

declare <2 x i64> @llvm.vp.load.v2i64.p0(ptr, <2 x i1>, i32)

declare void @readsMem(ptr) #0
declare void @writesMem(ptr) #1

attributes #0 = { memory(read) }
attributes #1 = { memory(write) }

!0 = !{!"scope", !"workgroup"}
!1 = !{!"as", !"private"}
!2 = !{!0, !1}
