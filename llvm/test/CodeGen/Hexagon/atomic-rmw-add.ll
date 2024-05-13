; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK: .LBB0_1:
; CHECK:      [[R1:r[0-9]+]] = memw_locked(r0)
; CHECK-DAG:  [[R2:r[0-9]+]] = and([[R1]],
; CHECK-DAG:  [[R3:r[0-9]+]] = add([[R1]],
; CHECK:      [[R2]] |= and([[R3]],
; CHECK:      memw_locked(r0,[[P0:p[0-3]]]) = [[R2]]
; CHECK:      if (![[P0]]) jump:nt .LBB0_1


%struct.a = type { i8 }

define void @b() #0 {
  %d = alloca %struct.a
  atomicrmw add ptr %d, i8 2 monotonic
  ret void
}

attributes #0 = { "target-cpu"="hexagonv66" }

