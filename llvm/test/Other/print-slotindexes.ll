; RUN: llc -print-after=slotindexes < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,SI 
; RUN: llc -print-after=slotindexes -print-slotindexes=false < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,NOSI
; REQUIRES: default_triple
define void @foo(){
  ret void
}

;CHECK: IR Dump {{.*}}
;CHECK: # Machine code for function foo{{.*}}
;SI: {{[0-9]+}}B bb.0 (%ir-block.0)
;NOSI: {{^}}bb.0 (%ir-block.0)

