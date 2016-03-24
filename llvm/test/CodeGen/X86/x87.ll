; RUN: llc < %s -march=x86 | FileCheck %s -check-prefix=X87
; RUN: llc < %s -march=x86-64 -mattr=-sse | FileCheck %s -check-prefix=X87
; RUN: llc < %s -march=x86 -mattr=-x87 | FileCheck %s -check-prefix=NOX87 --implicit-check-not "{{ }}f{{.*}}"
; RUN: llc < %s -march=x86-64 -mattr=-x87,-sse | FileCheck %s -check-prefix=NOX87 --implicit-check-not "{{ }}f{{.*}}"
; RUN: llc < %s -march=x86 -mattr=-x87,+sse | FileCheck %s -check-prefix=NOX87 --implicit-check-not "{{ }}f{{.*}}"
; RUN: llc < %s -march=x86-64 -mattr=-x87,-sse2 | FileCheck %s -check-prefix=NOX87 --implicit-check-not "{{ }}f{{.*}}"

define void @test(i32 %i, i64 %l, float* %pf, double* %pd, fp128* %pld) nounwind readnone {
; X87-LABEL: test:
; NOX87-LABEL: test:
; X87: fild
; NOX87: __floatunsisf
  %tmp = uitofp i32 %i to float

; X87: fild
; NOX87: __floatdisf
  %tmp1 = sitofp i64 %l to float

; X87: fadd
; NOX87: __addsf3
  %tmp2 = fadd float %tmp, %tmp1

; X87: fstp
  store float %tmp2, float* %pf

; X87: fild
; NOX87: __floatunsidf
  %tmp3 = uitofp i32 %i to double

; X87: fild
; NOX87: __floatdidf
  %tmp4 = sitofp i64 %l to double

; X87: fadd
; NOX87: __adddf3
  %tmp5 = fadd double %tmp3, %tmp4

; X87: fstp
  store double %tmp5, double* %pd

; X87: __floatsitf
; NOX87: __floatsitf
  %tmp6 = sitofp i32 %i to fp128

; X87: __floatunditf
; NOX87: __floatunditf
  %tmp7 = uitofp i64 %l to fp128

; X87: __addtf3
; NOX87: __addtf3
  %tmp8 = fadd fp128 %tmp6, %tmp7
  store fp128 %tmp8, fp128* %pld

  ret void
}
