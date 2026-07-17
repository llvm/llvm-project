; RUN: opt %s -passes='sample-profile' -sample-profile-file=%S/Inputs/weak-symbol-profile-mismatch.prof -min-functions-for-staleness-error=1 2>&1 | FileCheck %s

; CHECK-NOT: Pseudo-probe-based profile requires SampleProfileProbePass
; Function Attrs: noinline nounwind optnone uwtable
define weak dso_local void @main() #0 align 8 {
  ret void
}

attributes #0 = { noinline nounwind optnone uwtable }

!llvm.pseudo_probe_desc = !{!1}
!1 = !{i64 -2624081020897602054, i64 123456, !"main"}
