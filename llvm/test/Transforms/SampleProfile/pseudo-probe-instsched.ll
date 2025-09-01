; REQUIRES: x86-registered-target
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-unknown-linux-gnu -O3 | FileCheck %s
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-unknown-windows-msvc -O3 | FileCheck %s

define float @foo(float %x) #0 {
  %tmp1 = fmul float %x, 3.000000e+00
  %tmp3 = fmul float %x, 5.000000e+00
  %tmp5 = fmul float %x, 7.000000e+00
  %tmp7 = fmul float %x, 1.100000e+01
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 1, i32 0, i64 -1)
  %tmp10 = fadd float %tmp1, %tmp3
  %tmp12 = fadd float %tmp10, %tmp5
  %tmp14 = fadd float %tmp12, %tmp7
  ret float %tmp14
; CHECK: mulss
; CHECK: mulss
; CHECK: addss
; CHECK: mulss
; CHECK: addss
; CHECK: mulss
; CHECK: addss
; CHECK: ret
}

; Function Attrs: inaccessiblememonly nounwind willreturn
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #1

attributes #0 = { nounwind }
attributes #1 = { inaccessiblememonly nounwind willreturn }

!llvm.pseudo_probe_desc = !{!0}

!0 = !{i64 6699318081062747564, i64 4294967295, !"foo", null}

