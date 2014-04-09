; RUN: llc < %s -O0 -mcpu=ppc64 | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define void @_ZN4llvm3sys17RunningOnValgrindEv() #0 {
entry:
  br i1 undef, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  ret void

if.end:                                           ; preds = %entry
  %0 = call i64 asm sideeffect "mr 3,$1\0A\09mr 4,$2\0A\09rotldi 0,0,3  ; rotldi 0,0,13\0A\09rotldi 0,0,61 ; rotldi 0,0,51\0A\09or 1,1,1\0A\09mr $0,3", "=b,b,b,~{cc},~{memory},~{r3},~{r4}"(i32 0, i64* undef) #0
  unreachable

; CHECK-LABEL: @_ZN4llvm3sys17RunningOnValgrindEv
; CHECK: stw
; CHECK: lwz
}

attributes #0 = { nounwind }

