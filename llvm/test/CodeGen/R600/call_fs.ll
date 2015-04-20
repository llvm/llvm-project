
; RUN: llc < %s -march=r600 -mcpu=redwood -show-mc-encoding -o - | FileCheck --check-prefix=EG %s
; RUN: llc < %s -march=r600 -mcpu=rv710 -show-mc-encoding -o - | FileCheck --check-prefix=R600 %s

; EG: {{^}}call_fs:
; EG: .long 257
; EG: CALL_FS  ; encoding: [0x00,0x00,0x00,0x00,0x00,0x00,0xc0,0x84]
; R600: {{^}}call_fs:
; R600: .long 257
; R600:CALL_FS ; encoding: [0x00,0x00,0x00,0x00,0x00,0x00,0x80,0x89]


define void @call_fs() #0 {
  ret void
}

attributes #0 = { "ShaderType"="1" } ; Vertex Shader
