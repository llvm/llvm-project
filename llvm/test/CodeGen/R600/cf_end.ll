; RUN: llc < %s -march=r600 -mcpu=redwood --show-mc-encoding | FileCheck --check-prefix=EG %s
; RUN: llc < %s -march=r600 -mcpu=caicos --show-mc-encoding | FileCheck --check-prefix=EG %s
; RUN: llc < %s -march=r600 -mcpu=cayman --show-mc-encoding | FileCheck --check-prefix=CM %s

; EG: CF_END ; encoding: [0x00,0x00,0x00,0x00,0x00,0x00,0x20,0x80]
; CM: CF_END ; encoding: [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x88]
define void @eop() {
  ret void
}
