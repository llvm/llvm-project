; RUN: llvm-mc -triple msp430 -show-encoding %s \
; RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

; RUN: llvm-mc -triple msp430 -filetype=obj %s \
; RUN:     | llvm-objdump -d - | FileCheck --check-prefix=CHECK-INST %s

  ;; IForm8 instructions
  mov.b  r7, r8 ; CHECK-INST: mov.b  r7, r8
                ; CHECK: encoding: [0x48,0x47]
  add.b  r7, r8 ; CHECK-INST: add.b  r7, r8
                ; CHECK: encoding: [0x48,0x57]
  addc.b r7, r8 ; CHECK-INST: addc.b r7, r8
                ; CHECK: encoding: [0x48,0x67]
  subc.b r7, r8 ; CHECK-INST: subc.b r7, r8
                ; CHECK: encoding: [0x48,0x77]
  sub.b  r7, r8 ; CHECK-INST: sub.b  r7, r8
                ; CHECK: encoding: [0x48,0x87]
  cmp.b  r7, r8 ; CHECK-INST: cmp.b  r7, r8
                ; CHECK: encoding: [0x48,0x97]
  dadd.b r7, r8 ; CHECK-INST: dadd.b r7, r8
                ; CHECK: encoding: [0x48,0xa7]
  bit.b  r7, r8 ; CHECK-INST: bit.b  r7, r8
                ; CHECK: encoding: [0x48,0xb7]
  bic.b  r7, r8 ; CHECK-INST: bic.b  r7, r8
                ; CHECK: encoding: [0x48,0xc7]
  bis.b  r7, r8 ; CHECK-INST: bis.b  r7, r8
                ; CHECK: encoding: [0x48,0xd7]
  xor.b  r7, r8 ; CHECK-INST: xor.b  r7, r8
                ; CHECK: encoding: [0x48,0xe7]
  and.b  r7, r8 ; CHECK-INST: and.b  r7, r8
                ; CHECK: encoding: [0x48,0xf7]

  ;; IForm16 instructions
  mov    r7, r8 ; CHECK-INST: mov    r7, r8
                ; CHECK: encoding: [0x08,0x47]
  add    r7, r8 ; CHECK-INST: add    r7, r8
                ; CHECK: encoding: [0x08,0x57]
  addc   r7, r8 ; CHECK-INST: addc   r7, r8
                ; CHECK: encoding: [0x08,0x67]
  subc   r7, r8 ; CHECK-INST: subc   r7, r8
                ; CHECK: encoding: [0x08,0x77]
  sub    r7, r8 ; CHECK-INST: sub    r7, r8
                ; CHECK: encoding: [0x08,0x87]
  cmp    r7, r8 ; CHECK-INST: cmp    r7, r8
                ; CHECK: encoding: [0x08,0x97]
  dadd   r7, r8 ; CHECK-INST: dadd   r7, r8
                ; CHECK: encoding: [0x08,0xa7]
  bit    r7, r8 ; CHECK-INST: bit    r7, r8
                ; CHECK: encoding: [0x08,0xb7]
  bic    r7, r8 ; CHECK-INST: bic    r7, r8
                ; CHECK: encoding: [0x08,0xc7]
  bis    r7, r8 ; CHECK-INST: bis    r7, r8
                ; CHECK: encoding: [0x08,0xd7]
  xor    r7, r8 ; CHECK-INST: xor    r7, r8
                ; CHECK: encoding: [0x08,0xe7]
  and    r7, r8 ; CHECK-INST: and    r7, r8
                ; CHECK: encoding: [0x08,0xf7]

  ;; IIForm8 instructions
  rrc.b  r7     ; CHECK-INST: rrc.b  r7    
                ; CHECK: encoding: [0x47,0x10]
  rra.b  r7     ; CHECK-INST: rra.b  r7    
                ; CHECK: encoding: [0x47,0x11]
  push.b r7     ; CHECK-INST: push.b r7    
                ; CHECK: encoding: [0x47,0x12]

  ;; IIForm16 instructions
  rrc    r7     ; CHECK-INST: rrc    r7    
                ; CHECK: encoding: [0x07,0x10]
  swpb   r7     ; CHECK-INST: swpb   r7    
                ; CHECK: encoding: [0x87,0x10]
  rra    r7     ; CHECK-INST: rra    r7    
                ; CHECK: encoding: [0x07,0x11]
  sxt    r7     ; CHECK-INST: sxt    r7    
                ; CHECK: encoding: [0x87,0x11]
  push   r7     ; CHECK-INST: push   r7    
                ; CHECK: encoding: [0x07,0x12]
  call   r7     ; CHECK-INST: call   r7    
                ; CHECK: encoding: [0x87,0x12]
  reti          ; CHECK-INST: reti         
                ; CHECK: encoding: [0x00,0x13]

  ;; CJForm instructions
  jnz    -2     ; CHECK-INST: jne    $-2
                ; CHECK: encoding: [0xfe,0x23]
  jne    -2     ; CHECK-INST: jne    $-2
                ; CHECK: encoding: [0xfe,0x23]
  jeq    -2     ; CHECK-INST: jeq    $-2
                ; CHECK: encoding: [0xfe,0x27]
  jz     -2     ; CHECK-INST: jeq    $-2
                ; CHECK: encoding: [0xfe,0x27]
  jnc    -2     ; CHECK-INST: jlo    $-2
                ; CHECK: encoding: [0xfe,0x2b]
  jlo    -2     ; CHECK-INST: jlo    $-2
                ; CHECK: encoding: [0xfe,0x2b]
  jc     -2     ; CHECK-INST: jhs    $-2
                ; CHECK: encoding: [0xfe,0x2f]
  jhs    -2     ; CHECK-INST: jhs    $-2
                ; CHECK: encoding: [0xfe,0x2f]
  jn     -2     ; CHECK-INST: jn     $-2
                ; CHECK: encoding: [0xfe,0x33]
  jge    -2     ; CHECK-INST: jge    $-2
                ; CHECK: encoding: [0xfe,0x37]
  jl     -2     ; CHECK-INST: jl     $-2
                ; CHECK: encoding: [0xfe,0x3b]
  jmp    $-2    ; CHECK-INST: jmp    $-2
                ; CHECK: encoding: [0xfe,0x3f]

  ;; Emulated arithmetic instructions
  adc    r7     ; CHECK-INST: adc    r7
                ; CHECK: encoding: [0x07,0x63]
  dadc   r7     ; CHECK-INST: dadc   r7
                ; CHECK: encoding: [0x07,0xa3]
  dec    r7     ; CHECK-INST: dec    r7
                ; CHECK: encoding: [0x17,0x83]
  decd   r7     ; CHECK-INST: decd   r7
                ; CHECK: encoding: [0x27,0x83]
  inc    r7     ; CHECK-INST: inc    r7
                ; CHECK: encoding: [0x17,0x53]
  incd   r7     ; CHECK-INST: incd   r7
                ; CHECK: encoding: [0x27,0x53]
  sbc    r7     ; CHECK-INST: sbc    r7
                ; CHECK: encoding: [0x07,0x73]

  ;; Emulated logical instructions
  inv    r7     ; CHECK-INST: inv    r7
                ; CHECK: encoding: [0x37,0xe3]
  rla    r7     ; CHECK-INST: add    r7, r7
                ; CHECK: encoding: [0x07,0x57]
  rlc    r7     ; CHECK-INST: addc   r7, r7
                ; CHECK: encoding: [0x07,0x67]

  ;; Emulated program flow control instructions
  br     r7     ; CHECK-INST: br     r7    
                ; CHECK: encoding: [0x00,0x47]
  dint          ; CHECK-INST: dint
                ; CHECK: encoding: [0x32,0xc2]
  eint          ; CHECK-INST: eint
                ; CHECK: encoding: [0x32,0xd2]
  nop           ; CHECK-INST: nop
                ; CHECK: encoding: [0x03,0x43]
  ret           ; CHECK-INST: ret          
                ; CHECK: encoding: [0x30,0x41]

  ;; Emulated data instruction
  clr    r7     ; CHECK-INST: clr    r7
                ; CHECK: encoding: [0x07,0x43]
  clrc          ; CHECK-INST: clrc
                ; CHECK: encoding: [0x12,0xc3]
  clrn          ; CHECK-INST: clrn
                ; CHECK: encoding: [0x22,0xc2]
  clrz          ; CHECK-INST: clrz
                ; CHECK: encoding: [0x22,0xc3]
  pop    r7     ; CHECK-INST: pop    r7
                ; CHECK: encoding: [0x37,0x41]
  setc          ; CHECK-INST: setc
                ; CHECK: encoding: [0x12,0xd3]
  setn          ; CHECK-INST: setn
                ; CHECK: encoding: [0x22,0xd2]
  setz          ; CHECK-INST: setz
                ; CHECK: encoding: [0x22,0xd3]
  tst    r7     ; CHECK-INST: tst    r7
                ; CHECK: encoding: [0x07,0x93]
