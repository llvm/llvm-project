; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

target triple = "hexagon"

%struct.0 = type { i64, i16 }

declare void @foo(ptr noalias nocapture sret(%struct.0), i8 zeroext, i32, i64) #0

define hidden fastcc void @fred(ptr noalias nocapture %p, i8 zeroext %t, i32 %r) unnamed_addr #0 {
entry:
  %. = select i1 undef, i64 549755813888, i64 1024
  %cmp104 = icmp ult i64 undef, %.
  %inc = zext i1 %cmp104 to i32
  %inc.r = add nsw i32 %inc, %r
  %.inc.r = select i1 undef, i32 0, i32 %inc.r
  tail call void @foo(ptr sret(%struct.0) %p, i8 zeroext %t, i32 %.inc.r, i64 undef)
  ret void
}

attributes #0 = { noinline nounwind }

