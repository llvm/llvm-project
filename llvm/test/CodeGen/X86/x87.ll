; RUN: llc < %s -mtriple=i686-- | FileCheck %s -check-prefixes=X8732,X87
; RUN: llc < %s -mtriple=x86_64-- -mattr=-sse | FileCheck %s -check-prefixes=X8732,X87
; RUN: llc < %s -mtriple=i686-- -mattr=-x87 | FileCheck %s -check-prefixes=NOX8732,NOX87
; RUN: llc < %s -mtriple=x86_64-- -mattr=-x87,-sse | FileCheck %s -check-prefixes=NOX8732,NOX87
; RUN: llc < %s -mtriple=i686-- -mattr=-x87,+sse | FileCheck %s -check-prefixes=NOX8732,NOX87
; RUN: llc < %s -mtriple=x86_64-- -mattr=-x87,-sse2 | FileCheck %s -check-prefixes=NOX87

define void @test(i32 %i, i64 %l, ptr %pf, ptr %pd, ptr %pld) nounwind readnone {
; X87-LABEL: test:
; NOX87-LABEL: test:

; NOX87-NOT: {{ }}f{{.*}}

; X87: fild
; NOX8732: __floatunsisf
  %tmp = uitofp i32 %i to float

; X8732: fild
; NOX8732: __floatdisf
  %tmp1 = sitofp i64 %l to float

; X8732: fadd
; NOX8732: __addsf3
  %tmp2 = fadd float %tmp, %tmp1

; X8732: fstp
  store float %tmp2, ptr %pf

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
  store double %tmp5, ptr %pd

; X87: __floatsitf
; NOX87: __floatsitf
  %tmp6 = sitofp i32 %i to fp128

; X87: __floatunditf
; NOX87: __floatunditf
  %tmp7 = uitofp i64 %l to fp128

; X87: __addtf3
; NOX87: __addtf3
  %tmp8 = fadd fp128 %tmp6, %tmp7
  store fp128 %tmp8, ptr %pld

  ret void
}
