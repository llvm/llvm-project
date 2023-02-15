; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -d - | FileCheck --check-prefix=INST %s


foo:
  ; BREQ
  breq .-18
  breq .-12
  brbs 1, .-18
  brbs 1, baz

; CHECK: breq    .Ltmp0-18               ; encoding: [0bAAAAA001,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp0-18, kind: fixup_7_pcrel
; CHECK: breq    .Ltmp1-12               ; encoding: [0bAAAAA001,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp1-12, kind: fixup_7_pcrel
; CHECK: brbs    1, .Ltmp2-18            ; encoding: [0bAAAAA001,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp2-18, kind: fixup_7_pcrel
; CHECK: brbs    1, baz                  ; encoding: [0bAAAAA001,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: baz, kind: fixup_7_pcrel

; INST-LABEL: <foo>:
; INST: breq .-18
; INST: breq .-12
; INST: breq .-18
; INST: breq .+0

  ; BRNE
  brne .+10
  brne .+2
  brbc 1, .+10
  brbc 1, bar

; CHECK: brne    .Ltmp3+10               ; encoding: [0bAAAAA001,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp3+10, kind: fixup_7_pcrel
; CHECK: brne    .Ltmp4+2                ; encoding: [0bAAAAA001,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp4+2, kind: fixup_7_pcrel
; CHECK: brbc    1, .Ltmp5+10            ; encoding: [0bAAAAA001,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp5+10, kind: fixup_7_pcrel
; CHECK: brbc    1, bar                  ; encoding: [0bAAAAA001,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: bar, kind: fixup_7_pcrel

; INST: brne .+10
; INST: brne .+2
; INST: brne .+10
; INST: brne .+0

bar:
  ; BRCS
  brcs .+8
  brcs .+4
  brbs 0, .+8
  brbs 0, end

; CHECK: brcs    .Ltmp6+8                ; encoding: [0bAAAAA000,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp6+8, kind: fixup_7_pcrel
; CHECK: brcs    .Ltmp7+4                ; encoding: [0bAAAAA000,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp7+4, kind: fixup_7_pcrel
; CHECK: brcs    .Ltmp8+8                ; encoding: [0bAAAAA000,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp8+8, kind: fixup_7_pcrel
; CHECK: brcs    end                     ; encoding: [0bAAAAA000,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: end, kind: fixup_7_pcrel

; INST-LABEL: <bar>:
; INST: brlo .+8
; INST: brlo .+4
; INST: brlo .+8
; INST: brlo .+0

  ; BRCC
  brcc .+66
  brcc .-22
  brbc 0, .+66
  brbc 0, baz

; CHECK: brcc    .Ltmp9+66               ; encoding: [0bAAAAA000,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp9+66, kind: fixup_7_pcrel
; CHECK: brcc    .Ltmp10-22              ; encoding: [0bAAAAA000,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp10-22, kind: fixup_7_pcrel
; CHECK: brcc    .Ltmp11+66              ; encoding: [0bAAAAA000,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp11+66, kind: fixup_7_pcrel
; CHECK: brcc    baz                     ; encoding: [0bAAAAA000,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: baz, kind: fixup_7_pcrel

; INST: brsh .+66
; INST: brsh .-22
; INST: brsh .+66
; INST: brsh .+0

; BRSH
  brsh .+32
  brsh .+70
  brsh car

; CHECK: brsh    .Ltmp12+32              ; encoding: [0bAAAAA000,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp12+32, kind: fixup_7_pcrel
; CHECK: brsh    .Ltmp13+70              ; encoding: [0bAAAAA000,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp13+70, kind: fixup_7_pcrel
; CHECK: brsh    car                     ; encoding: [0bAAAAA000,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: car, kind: fixup_7_pcrel

; INST: brsh .+32
; INST: brsh .+70
; INST: brsh .+0

baz:

  ; BRLO
  brlo .+12
  brlo .+28
  brlo car

; CHECK: brlo    .Ltmp14+12              ; encoding: [0bAAAAA000,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp14+12, kind: fixup_7_pcrel
; CHECK: brlo    .Ltmp15+28              ; encoding: [0bAAAAA000,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp15+28, kind: fixup_7_pcrel
; CHECK: brlo    car                     ; encoding: [0bAAAAA000,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: car, kind: fixup_7_pcrel

; INST-LABEL: <baz>:
; INST: brlo .+12
; INST: brlo .+28
; INST: brlo .+0

  ; BRMI
  brmi .+66
  brmi .+58
  brmi car

; CHECK: brmi    .Ltmp16+66              ; encoding: [0bAAAAA010,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp16+66, kind: fixup_7_pcrel
; CHECK: brmi    .Ltmp17+58              ; encoding: [0bAAAAA010,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp17+58, kind: fixup_7_pcrel
; CHECK: brmi    car                     ; encoding: [0bAAAAA010,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: car, kind: fixup_7_pcrel

; INST: brmi .+66
; INST: brmi .+58
; INST: brmi .+0

  ; BRPL
  brpl .-12
  brpl .+18
  brpl car

; CHECK: brpl    .Ltmp18-12              ; encoding: [0bAAAAA010,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp18-12, kind: fixup_7_pcrel
; CHECK: brpl    .Ltmp19+18              ; encoding: [0bAAAAA010,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp19+18, kind: fixup_7_pcrel
; CHECK: brpl    car                     ; encoding: [0bAAAAA010,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: car, kind: fixup_7_pcrel

; INST: brpl .-12
; INST: brpl .+18
; INST: brpl .+0

; BRGE
  brge .+50
  brge .+42
  brge car

; CHECK: brge    .Ltmp20+50              ; encoding: [0bAAAAA100,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp20+50, kind: fixup_7_pcrel
; CHECK: brge    .Ltmp21+42              ; encoding: [0bAAAAA100,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp21+42, kind: fixup_7_pcrel
; CHECK: brge    car                     ; encoding: [0bAAAAA100,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: car, kind: fixup_7_pcrel

; INST: brge .+50
; INST: brge .+42
; INST: brge .+0

car:
  ; BRLT
  brlt .+16
  brlt .+2
  brlt end

; CHECK: brlt    .Ltmp22+16              ; encoding: [0bAAAAA100,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp22+16, kind: fixup_7_pcrel
; CHECK: brlt    .Ltmp23+2               ; encoding: [0bAAAAA100,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp23+2, kind: fixup_7_pcrel
; CHECK: brlt    end                     ; encoding: [0bAAAAA100,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: end, kind: fixup_7_pcrel

; INST-LABEL: <car>:
; INST: brlt	.+16
; INST: brlt	.+2
; INST: brlt	.+0

  ; BRHS
  brhs .-66
  brhs .+14
  brhs just_another_label

; CHECK: brhs    .Ltmp24-66              ; encoding: [0bAAAAA101,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp24-66, kind: fixup_7_pcrel
; CHECK: brhs    .Ltmp25+14              ; encoding: [0bAAAAA101,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp25+14, kind: fixup_7_pcrel
; CHECK: brhs    just_another_label      ; encoding: [0bAAAAA101,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: just_another_label, kind: fixup_7_pcrel

; INST: brhs	.-66
; INST: brhs	.+14
; INST: brhs	.+0

  ; BRHC
  brhc .+12
  brhc .+14
  brhc just_another_label

; CHECK: brhc    .Ltmp26+12              ; encoding: [0bAAAAA101,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp26+12, kind: fixup_7_pcrel
; CHECK: brhc    .Ltmp27+14              ; encoding: [0bAAAAA101,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp27+14, kind: fixup_7_pcrel
; CHECK: brhc    just_another_label      ; encoding: [0bAAAAA101,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: just_another_label, kind: fixup_7_pcrel

; INST: brhc	.+12
; INST: brhc	.+14
; INST: brhc	.+0

  ; BRTS
  brts .+18
  brts .+22
  brts just_another_label

; CHECK: brts    .Ltmp28+18              ; encoding: [0bAAAAA110,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp28+18, kind: fixup_7_pcrel
; CHECK: brts    .Ltmp29+22              ; encoding: [0bAAAAA110,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp29+22, kind: fixup_7_pcrel
; CHECK: brts    just_another_label      ; encoding: [0bAAAAA110,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: just_another_label, kind: fixup_7_pcrel

; INST: brts	.+18
; INST: brts	.+22
; INST: brts	.+0

just_another_label:
  ; BRTC
  brtc .+52
  brtc .+50
  brtc end

; CHECK: brtc    .Ltmp30+52              ; encoding: [0bAAAAA110,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp30+52, kind: fixup_7_pcrel
; CHECK: brtc    .Ltmp31+50              ; encoding: [0bAAAAA110,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp31+50, kind: fixup_7_pcrel
; CHECK: brtc    end                     ; encoding: [0bAAAAA110,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: end, kind: fixup_7_pcrel

; INST-LABEL: <just_another_label>:
; INST: brtc	.+52
; INST: brtc	.+50
; INST: brtc	.+0

  ; BRVS
  brvs .+18
  brvs .+32
  brvs end

; CHECK: brvs    .Ltmp32+18              ; encoding: [0bAAAAA011,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp32+18, kind: fixup_7_pcrel
; CHECK: brvs    .Ltmp33+32              ; encoding: [0bAAAAA011,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp33+32, kind: fixup_7_pcrel
; CHECK: brvs    end                     ; encoding: [0bAAAAA011,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: end, kind: fixup_7_pcrel

; INST: brvs	.+18
; INST: brvs	.+32
; INST: brvs	.+0

  ; BRVC
  brvc .-28
  brvc .-62
  brvc end

; CHECK: brvc    .Ltmp34-28              ; encoding: [0bAAAAA011,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp34-28, kind: fixup_7_pcrel
; CHECK: brvc    .Ltmp35-62              ; encoding: [0bAAAAA011,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp35-62, kind: fixup_7_pcrel
; CHECK: brvc    end                     ; encoding: [0bAAAAA011,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: end, kind: fixup_7_pcrel

; INST: brvc	.-28
; INST: brvc	.-62
; INST: brvc	.+0

  ; BRIE
  brie .+20
  brie .+40
  brie end

; CHECK: brie    .Ltmp36+20              ; encoding: [0bAAAAA111,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp36+20, kind: fixup_7_pcrel
; CHECK: brie    .Ltmp37+40              ; encoding: [0bAAAAA111,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp37+40, kind: fixup_7_pcrel
; CHECK: brie    end                     ; encoding: [0bAAAAA111,0b111100AA]
; CHECK:                                 ;   fixup A - offset: 0, value: end, kind: fixup_7_pcrel

; INST: brie	.+20
; INST: brie	.+40
; INST: brie	.+0

  ; BRID
  brid .+42
  brid .+62
  brid end

; CHECK: brid    .Ltmp38+42              ; encoding: [0bAAAAA111,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp38+42, kind: fixup_7_pcrel
; CHECK: brid    .Ltmp39+62              ; encoding: [0bAAAAA111,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: .Ltmp39+62, kind: fixup_7_pcrel
; CHECK: brid    end                     ; encoding: [0bAAAAA111,0b111101AA]
; CHECK:                                 ;   fixup A - offset: 0, value: end, kind: fixup_7_pcrel

; INST: brid	.+42
; INST: brid	.+62
; INST: brid	.+0

end:
