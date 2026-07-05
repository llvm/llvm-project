
; RUN: llc < %s -mtriple=r600 -mcpu=redwood -show-mc-encoding -o - | FileCheck --check-prefix=EG %s
; RUN: llc < %s -mtriple=r600 -mcpu=rv710 -show-mc-encoding -o - | FileCheck --check-prefix=R600 %s

; EG: .long 257
; EG: {{^}}call_fs:
; EG: CALL_FS  ; encoding: [0x00,0x00,0x00,0x00,0x00,0x00,0xc0,0x84]
; R600: .long 257
; R600: {{^}}call_fs:
; R600:CALL_FS ; encoding: [0x00,0x00,0x00,0x00,0x00,0x00,0x80,0x89]


define amdgpu_vs void @call_fs() {
  ret void
}
