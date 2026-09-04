; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-ipra \
; RUN:     -print-after=DummyCGSCCPass -o - %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-ipra \
; RUN:     -print-after=DummyCGSCCPass -print-module-scope -o - %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=PERSISTENT
; REQUIRES: x86-registered-target

; The legacy CallGraphSCCPass printer should emit the banner as its own line.
; CHECK-LABEL: *** IR Dump After DummyCGSCCPass (DummyCGSCCPass) ***
; CHECK-NEXT: define void @bar() {

; PERSISTENT-LABEL: *** IR Dump After DummyCGSCCPass (DummyCGSCCPass) ***
; PERSISTENT: define void @bar() {
; PERSISTENT: ret void, !annotation ![[USED:[0-9]+]]
; PERSISTENT: ![[USED]] = !{!"used"}

define void @bar() {
  ret void, !annotation !1
}

!0 = !{!"unused"}
!1 = !{!"used"}
