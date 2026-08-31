; Verify that the AMDGPU summary table keeps the prevailing definition's entry.
; RUN: split-file %s %t
; RUN: opt -module-summary < %t/a.ll -o %t/a.bc
; RUN: opt -module-summary < %t/b.ll -o %t/b.bc
; RUN: llvm-lto2 run %t/a.bc %t/b.bc -o %t/o \
; RUN:   -thinlto-distributed-indexes -r=%t/a.bc,f, -r=%t/b.bc,f,px
; RUN: llvm-bcanalyzer -dump %t/a.bc.thinlto.bc | FileCheck %s
; RUN: llvm-bcanalyzer -dump %t/b.bc.thinlto.bc | FileCheck %s

; CHECK:     <GLOBALVAL_SUMMARY_BLOCK
; CHECK-NOT:   <AMDGPU_ENTRY {{.*}} op4=2
; CHECK:       <AMDGPU_ENTRY op0=0 op1=2 op2=0 op3=0 op4=7 op5=0 op6=0 op7=0 op8=0/>
; CHECK-NOT:   <AMDGPU_ENTRY
; CHECK:     </GLOBALVAL_SUMMARY_BLOCK>

;--- a.ll
target triple = "amdgcn-amd-amdhsa"

define weak_odr void @f() #0 {
  ret void
}

attributes #0 = { "amdgpu-waves-per-eu"="2" }

;--- b.ll
target triple = "amdgcn-amd-amdhsa"

define weak_odr void @f() #0 {
  ret void
}

attributes #0 = { "amdgpu-waves-per-eu"="7" }
