; RUN: llc -mtriple=x86_64-pc-linux -stop-after=slotindexes %s -o - | llc -passes='print<slot-indexes>' -x mir 2>&1 | FileCheck %s

define void @foo(){
  ret void
}

; CHECK: Slot indexes in machine function: foo
; CHECK: 0
; CHECK: 16 RET64
; CHECK: 32
; CHECK: %bb.0	[0B;32B)
