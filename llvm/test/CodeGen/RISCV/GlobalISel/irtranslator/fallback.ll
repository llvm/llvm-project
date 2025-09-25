; RUN: llc -mtriple=riscv64 -mattr='+v' -O0 -global-isel -global-isel-abort=2 -pass-remarks-missed='gisel*' -verify-machineinstrs %s -o %t.out 2> %t.err
; RUN: FileCheck %s --check-prefix=FALLBACK-WITH-REPORT-OUT < %t.out
; RUN: FileCheck %s --check-prefix=FALLBACK-WITH-REPORT-ERR < %t.err

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to translate instruction: alloca:
; FALLBACK-WITH-REPORT-OUT-LABEL: scalable_alloca
define void @scalable_alloca() #1 {
  %local0 = alloca <vscale x 16 x i8>
  load volatile <vscale x 16 x i8>, ptr %local0
  ret void
}
