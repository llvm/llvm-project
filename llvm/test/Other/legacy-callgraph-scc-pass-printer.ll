; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-ipra \
; RUN:     -print-after=DummyCGSCCPass -o - %s 2>&1 | FileCheck %s
; REQUIRES: x86-registered-target

; The legacy CallGraphSCCPass printer should emit the banner as its own line.
; CHECK-LABEL: *** IR Dump After DummyCGSCCPass (DummyCGSCCPass) ***
; CHECK-NEXT: define void @bar() {

define void @bar() {
  ret void
}
