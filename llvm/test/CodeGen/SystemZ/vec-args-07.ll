; Test calling functions with multiple return values (LLVM ABI extension)
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 -mattr=soft-float \
; RUN:   | FileCheck %s --check-prefix=SOFT-FLOAT

; Up to eight vector return values fit into VRs.
declare { <2 x double>, <2 x double>, <2 x double>, <2 x double>,
          <2 x double>, <2 x double>, <2 x double>, <2 x double> } @bar1()

define <2 x double> @f1() {
; CHECK-LABEL: f1:
; CHECK: brasl %r14, bar1
; CHECK: vlr %v24, %v31
; CHECK: br %r14

; SOFT-FLOAT-LABEL: f1:
; SOFT-FLOAT-NOT: %{{[fv]}}
; SOFT-FLOAT: brasl   %r14, bar1
; SOFT-FLOAT-NEXT: lg      %r3, 280(%r15)
; SOFT-FLOAT-NEXT: lg      %r2, 272(%r15)
; SOFT-FLOAT-NEXT: lmg     %r14, %r15, 400(%r15)
; SOFT-FLOAT-NEXT: br      %r14
  %mret = call { <2 x double>, <2 x double>,
                 <2 x double>, <2 x double>,
                 <2 x double>, <2 x double>,
                 <2 x double>, <2 x double> } @bar1()
  %ret = extractvalue { <2 x double>, <2 x double>,
                        <2 x double>, <2 x double>,
                        <2 x double>, <2 x double>,
                        <2 x double>, <2 x double> } %mret, 7
  ret <2 x double> %ret
}

; More than eight vector return values use sret.
declare { <2 x double>, <2 x double>, <2 x double>, <2 x double>,
          <2 x double>, <2 x double>, <2 x double>, <2 x double>,
          <2 x double> } @bar2()

define <2 x double> @f2() {
; CHECK-LABEL: f2:
; CHECK: la %r2, 160(%r15)
; CHECK: brasl %r14, bar2
; CHECK: vl  %v24, 288(%r15)
; CHECK: br %r14

; SOFT-FLOAT-LABEL: f2:
; SOFT-FLOAT-NOT: %{{[fv]}}
; SOFT-FLOAT: brasl   %r14, bar2
; SOFT-FLOAT-NEXT: lg      %r3, 296(%r15)
; SOFT-FLOAT-NEXT: lg      %r2, 288(%r15)
; SOFT-FLOAT-NEXT: lmg     %r14, %r15, 416(%r15)
; SOFT-FLOAT-NEXT: br      %r14
  %mret = call { <2 x double>, <2 x double>,
                 <2 x double>, <2 x double>,
                 <2 x double>, <2 x double>,
                 <2 x double>, <2 x double>,
                 <2 x double> } @bar2()
  %ret = extractvalue { <2 x double>, <2 x double>,
                        <2 x double>, <2 x double>,
                        <2 x double>, <2 x double>,
                        <2 x double>, <2 x double>,
                        <2 x double> } %mret, 8
  ret <2 x double> %ret
}
