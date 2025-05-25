; RUN: llc -print-after=slotindexes -print-mi-addrs < %s 2>&1 | FileCheck %s
; REQUIRES: default_triple

; CHECK: IR Dump {{.*}}
; CHECK: # Machine code for function foo{{.*}}

define void @foo() {
  ; CHECK: ; 0x
  ret void
}

