; RUN: llc -mtriple=aarch64-windows-msvc %s -o - | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu %s -o - | FileCheck %s

target datalayout = "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128"

%class.X = type { i8 }
%struct.B = type { i32 (...)** }

$"??_9B@@$BA@AA" = comdat any

; Function Attrs: noinline optnone
define linkonce_odr void @"??_9B@@$BA@AA"(%struct.B* %this, ...) #1 comdat align 2  {
entry:
  %this.addr = alloca %struct.B*, align 8
  store %struct.B* %this, %struct.B** %this.addr, align 8
  %this1 = load %struct.B*, %struct.B** %this.addr, align 8
  call void asm sideeffect "", "~{d0}"()
  %0 = bitcast %struct.B* %this1 to void (%struct.B*, ...)***
  %vtable = load void (%struct.B*, ...)**, void (%struct.B*, ...)*** %0, align 8
  %vfn = getelementptr inbounds void (%struct.B*, ...)*, void (%struct.B*, ...)** %vtable, i64 0
  %1 = load void (%struct.B*, ...)*, void (%struct.B*, ...)** %vfn, align 8
  musttail call void (%struct.B*, ...) %1(%struct.B* %this1, ...)
  ret void
                                                  ; No predecessors!
  ret void
}

attributes #1 = { noinline optnone "thunk" }

; CHECK: mov     v16.16b, v0.16b
; CHECK: ldr     x8, [x0]
; CHECK: ldr     x8, [x8]
; CHECK: mov     v0.16b, v16.16b
; CHECK: br      x8
