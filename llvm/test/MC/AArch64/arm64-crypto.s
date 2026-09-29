; RUN: llvm-mc -triple arm64-apple-darwin -mattr=crypto -show-encoding -output-asm-variant=1 < %s | FileCheck %s
; RUN: llvm-mc -triple arm64-apple-darwin -mattr='+crypto,+fuse-aes' -show-encoding -output-asm-variant=1 < %s | FileCheck %s

foo:
  aese.16b v0, v1
  aesd.16b v0, v1
  aesmc.16b v0, v1
  aesimc.16b v0, v1

  sha1c.4s q0, s1, v2
  sha1p.4s q0, s1, v2
  sha1m.4s q0, s1, v2
  sha1su0.4s v0, v1, v2
  sha256h.4s q0, q1, v2
  sha256h2.4s q0, q1, v2
  sha256su1.4s v0, v1, v2
  sha1h s0, s1
  sha1su1.4s v0, v1
  sha256su0.4s v0, v1

; CHECK: aese.16b v0, v1               ; encoding: [0x20,0x48,0x28,0x4e]
; CHECK: aesd.16b v0, v1               ; encoding: [0x20,0x58,0x28,0x4e]
; CHECK: aesmc.16b v0, v1              ; encoding: [0x20,0x68,0x28,0x4e]
; CHECK: aesimc.16b v0, v1             ; encoding: [0x20,0x78,0x28,0x4e]

; CHECK: sha1c.4s q0, s1, v2           ; encoding: [0x20,0x00,0x02,0x5e]
; CHECK: sha1p.4s q0, s1, v2           ; encoding: [0x20,0x10,0x02,0x5e]
; CHECK: sha1m.4s q0, s1, v2           ; encoding: [0x20,0x20,0x02,0x5e]
; CHECK: sha1su0.4s v0, v1, v2         ; encoding: [0x20,0x30,0x02,0x5e]
; CHECK: sha256h.4s q0, q1, v2         ; encoding: [0x20,0x40,0x02,0x5e]
; CHECK: sha256h2.4s q0, q1, v2        ; encoding: [0x20,0x50,0x02,0x5e]
; CHECK: sha256su1.4s v0, v1, v2       ; encoding: [0x20,0x60,0x02,0x5e]
; CHECK: sha1h s0, s1                  ; encoding: [0x20,0x08,0x28,0x5e]
; CHECK: sha1su1.4s v0, v1             ; encoding: [0x20,0x18,0x28,0x5e]
; CHECK: sha256su0.4s v0, v1           ; encoding: [0x20,0x28,0x28,0x5e]

  aese v2.16b, v3.16b
  aesd v5.16b, v7.16b
  aesmc v11.16b, v13.16b
  aesimc v17.16b, v19.16b

; CHECK: aese.16b v2, v3            ; encoding: [0x62,0x48,0x28,0x4e]
; CHECK: aesd.16b v5, v7            ; encoding: [0xe5,0x58,0x28,0x4e]
; CHECK: aesmc.16b v11, v13         ; encoding: [0xab,0x69,0x28,0x4e]
; CHECK: aesimc.16b v17, v19        ; encoding: [0x71,0x7a,0x28,0x4e]

  sha1c q23, s29, v3.4s
  sha1p q14, s15, v9.4s
  sha1m q2, s6, v5.4s
  sha1su0 v3.4s, v5.4s, v9.4s
  sha256h q2, q7, v18.4s
  sha256h2 q28, q18, v28.4s
  sha256su1 v4.4s, v5.4s, v9.4s
  sha1h s30, s0
  sha1su1 v10.4s, v21.4s
  sha256su0 v2.4s, v31.4s

; CHECK: sha1c.4s q23, s29, v3       ; encoding: [0xb7,0x03,0x03,0x5e]
; CHECK: sha1p.4s q14, s15, v9       ; encoding: [0xee,0x11,0x09,0x5e]
; CHECK: sha1m.4s q2, s6, v5         ; encoding: [0xc2,0x20,0x05,0x5e]
; CHECK: sha1su0.4s v3, v5, v9       ; encoding: [0xa3,0x30,0x09,0x5e]
; CHECK: sha256h.4s q2, q7, v18      ; encoding: [0xe2,0x40,0x12,0x5e]
; CHECK: sha256h2.4s q28, q18, v28   ; encoding: [0x5c,0x52,0x1c,0x5e]
; CHECK: sha256su1.4s v4, v5, v9     ; encoding: [0xa4,0x60,0x09,0x5e]
; CHECK: sha1h s30, s0               ; encoding: [0x1e,0x08,0x28,0x5e]
; CHECK: sha1su1.4s v10, v21         ; encoding: [0xaa,0x1a,0x28,0x5e]
; CHECK: sha256su0.4s v2, v31        ; encoding: [0xe2,0x2b,0x28,0x5e]
