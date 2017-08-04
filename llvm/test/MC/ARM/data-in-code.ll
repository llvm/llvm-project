;; RUN: llc -verify-machineinstrs \
;; RUN:   -mtriple=armv7-linux-gnueabi -filetype=obj %s -o - | \
;; RUN:   llvm-readobj -t | FileCheck -check-prefix=ARM %s

;; RUN: llc -verify-machineinstrs \
;; RUN:   -mtriple=thumbv7-linux-gnueabi -filetype=obj %s -o - | \
;; RUN:   llvm-readobj -t | FileCheck -check-prefix=TMB %s

;; Ensure that if a jump table is generated that it has Mapping Symbols
;; marking the data-in-code region.

define void @foo(i32* %ptr, i32 %b) nounwind ssp {
  %tmp = load i32, i32* %ptr, align 4
  switch i32 %tmp, label %exit [
    i32 0, label %bb0
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
  ]
bb0:
  store i32 %b, i32* %ptr, align 4
  br label %exit
bb1:
  store i32 1, i32* %ptr, align 4
  br label %exit
bb2:
  store i32 2, i32* %ptr, align 4
  br label %exit
bb3:
  store i32 3, i32* %ptr, align 4
  br label %exit
exit:
  ret void
}

;; ARM:        Symbol {
;; ARM:          Name: $a
;; ARM-NEXT:     Value: 0x0
;; ARM-NEXT:     Size: 0
;; ARM-NEXT:     Binding: Local
;; ARM-NEXT:     Type: None
;; ARM-NEXT:     Other:
;; ARM-NEXT:     Section: [[MIXED_SECT:[^ ]+]]

;; ARM:        Symbol {
;; ARM:          Name: $a
;; ARM-NEXT:     Value: 0x{{[0-9A-F]+}}
;; ARM-NEXT:     Size: 0
;; ARM-NEXT:     Binding: Local
;; ARM-NEXT:     Type: None
;; ARM-NEXT:     Other:
;; ARM-NEXT:     Section: [[MIXED_SECT]]

;; ARM:        Symbol {
;; ARM:          Name: $d
;; ARM-NEXT:     Value: 0x{{[0-9A-F]+}}
;; ARM-NEXT:     Size: 0
;; ARM-NEXT:     Binding: Local
;; ARM-NEXT:     Type: None
;; ARM-NEXT:     Other:
;; ARM-NEXT:     Section: [[MIXED_SECT]]

;; ARM-NOT:     ${{[atd]}}

;; TMB:        Symbol {
;; TMB:          Name: $d.1
;; TMB-NEXT:     Value: 0x{{[0-9A-F]+}}
;; TMB-NEXT:     Size: 0
;; TMB-NEXT:     Binding: Local
;; TMB-NEXT:     Type: None
;; TMB-NEXT:     Other:
;; TMB-NEXT:     Section: [[MIXED_SECT:[^ ]+]]

;; TMB:        Symbol {
;; TMB:          Name: $t
;; TMB-NEXT:     Value: 0x0
;; TMB-NEXT:     Size: 0
;; TMB-NEXT:     Binding: Local
;; TMB-NEXT:     Type: None
;; TMB-NEXT:     Other:
;; TMB-NEXT:     Section: [[MIXED_SECT]]

;; TMB:        Symbol {
;; TMB:          Name: $t
;; TMB-NEXT:     Value: 0x{{[0-9A-F]+}}
;; TMB-NEXT:     Size: 0
;; TMB-NEXT:     Binding: Local
;; TMB-NEXT:     Type: None
;; TMB-NEXT:     Other:
;; TMB-NEXT:     Section: [[MIXED_SECT]]


;; TMB-NOT:     ${{[atd]}}

