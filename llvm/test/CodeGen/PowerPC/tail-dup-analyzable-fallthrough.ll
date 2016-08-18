; RUN: llc -O2 < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; Check that the conditional return block of fmax_double3.exit was not
; duplicated into the if.then.i block
; CHECK: # %if.then.i
; CHECK: lxvd2x
; CHECK: stxvd2x
; CHECK-NOT: bclr
; CHECK: {{^}}.LBB{{[0-9_]+}}:
; CHECK-SAME: # %fmax_double3.exit
; CHECK: bclr
; CHECK: # %if.then
; Function Attrs: nounwind
define void @__fmax_double3_3D_exec(<2 x double>* %input6, i1 %bool1, i1 %bool2) #0 {
entry:
  br i1 %bool1, label %if.then.i, label %fmax_double3.exit

if.then.i:                                        ; preds = %entry
  store <2 x double> zeroinitializer, <2 x double>* %input6, align 32
  br label %fmax_double3.exit

fmax_double3.exit:                                ; preds = %if.then.i, %entry
  br i1 %bool2, label %if.then, label %do.end

if.then:                                          ; preds = %fmax_double3.exit
  unreachable

do.end:                                           ; preds = %fmax_double3.exit
  ret void
}

attributes #0 = { nounwind }
