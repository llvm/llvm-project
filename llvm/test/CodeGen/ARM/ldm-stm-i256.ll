; RUN: llc -mtriple=armv7--eabi -verify-machineinstrs < %s | FileCheck %s

; Check the way we schedule/merge a bunch of loads and stores.
; Originally test/CodeGen/ARM/2011-07-07-ScheduleDAGCrash.ll ; now
; being used as a test of optimizations related to ldm/stm.

; FIXME: We could merge more loads/stores with regalloc hints.
; FIXME: Fix scheduling so we don't have 16 live registers.

define void @f(i256* nocapture %a, i256* nocapture %b, i256* nocapture %cc, i256* nocapture %dd) nounwind uwtable noinline ssp {
entry:
  %c = load i256, i256* %cc
  %d = load i256, i256* %dd
  %add = add nsw i256 %c, %d
  store i256 %add, i256* %a, align 8
  %or = or i256 %c, 1606938044258990275541962092341162602522202993782792835301376
  %add6 = add nsw i256 %or, %d
  store i256 %add6, i256* %b, align 8
  ret void
  ; CHECK-DAG: ldm r2
  ; CHECK-DAG: ldr {{.*}}, [r3]
  ; CHECK-DAG: ldr {{.*}}, [r3, #4]
  ; CHECK-DAG: ldr {{.*}}, [r3, #8]
  ; CHECK-DAG: ldr {{.*}}, [r3, #12]
  ; CHECK-DAG: ldr {{.*}}, [r3, #16]
  ; CHECK-DAG: ldr {{.*}}, [r3, #20]
  ; CHECK-DAG: ldr {{.*}}, [r3, #24]
  ; CHECK-DAG: ldr {{.*}}, [r3, #28]
  ; CHECK-DAG: ldr {{.*}}, [r2, #20]
  ; CHECK-DAG: ldr {{.*}}, [r2, #24]
  ; CHECK-DAG: ldr {{.*}}, [r2, #28]
  ; CHECK-DAG: stm r0
  ; CHECK-DAG: str {{.*}}, [r0, #20]
  ; CHECK-DAG: str {{.*}}, [r0, #24]
  ; CHECK-DAG: str {{.*}}, [r0, #28]
  ; CHECK-DAG: stm r1
  ; CHECK-DAG: str {{.*}}, [r1, #20]
  ; CHECK-DAG: str {{.*}}, [r1, #24]
  ; CHECK-DAG: str {{.*}}, [r1, #28]
}
