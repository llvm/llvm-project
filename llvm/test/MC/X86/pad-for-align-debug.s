# RUN: llvm-mc -mcpu=skylake -filetype=obj -triple x86_64-apple-macos -x86-pad-for-align=true %S/Inputs/pad-align-with-debug.s | llvm-objdump -d - | FileCheck --check-prefix=CHECK --check-prefix=DEBUG %s
# RUN: llvm-mc -mcpu=skylake -filetype=obj -triple x86_64-apple-macos -x86-pad-for-align=true %S/Inputs/pad-align-without-debug.s | llvm-objdump -d - | FileCheck --check-prefix=CHECK --check-prefix=NODEBUG %s
# RUN: llvm-mc -mcpu=skylake -filetype=obj -triple x86_64-apple-macos %S/Inputs/pad-align-without-debug.s | llvm-objdump -d - | FileCheck --check-prefix=DEFAULT %s
# RUN: llvm-mc -mcpu=skylake -filetype=obj -triple x86_64-apple-macos %S/Inputs/pad-align-with-debug.s | llvm-objdump -d - | FileCheck --check-prefix=DEFAULT %s

; Test case to show that -x86-pad-for-align causes binary differences in the
; presence of debug locations. Inputs/pad-align-with-debug.s and
; Inputs/pad-align-without-debug.s are equivalent, modulo a single .loc, which
; cause the difference in the binary below. This should be fixed, before
; x86-pad-for-align=true becomes the default.

; Also see PR48742.


; CHECK-LABEL: 0000000000000000 <foo>:
; CHECK:            0: 48 89 f8                         movq    %rdi, %rax
; CHECK-NEXT:       3: 0f 1f 44 00 00                   nopl    (%rax,%rax)
; CHECK-NEXT:       8: 0f b6 10                         movzbl  (%rax), %edx
; CHECK-NEXT:       b: 80 fa 0a                         cmpb    $10, %dl
; CHECK-NEXT:       e: 74 30                            je  0x40 <foo+0x40>
; CHECK-NEXT:      10: 80 fa 64                         cmpb    $100, %dl
; CHECK-NEXT:      13: 74 31                            je  0x46 <foo+0x46>
; CHECK-NEXT:      15: 80 fa c8                         cmpb    $-56, %dl
; CHECK-NEXT:      18: 74 2c                            je  0x46 <foo+0x46>
; CHECK-NEXT:      1a: 80 fa 2c                         cmpb    $44, %dl
; CHECK-NEXT:      1d: 75 27                            jne 0x46 <foo+0x46>
; CHECK-NEXT:      1f: 90                               nop
; CHECK-NEXT:      20: 48 89 c2                         movq    %rax, %rdx
; CHECK-NEXT:      23: 48 ff c0                         incq    %rax
; CHECK-NEXT:      26: 80 38 1e                         cmpb    $30, (%rax)

; DEBUG-NEXT:      29: 75 f5                            jne 0x20 <foo+0x20>
; DEBUG-NEXT:      2b: 48 8d 42 02                      leaq    2(%rdx), %rax
; DEBUG-NEXT:      2f: 0f b6 10                         movzbl  (%rax), %edx
; DEBUG-NEXT:      32: 80 fa 5a                         cmpb    $90, %dl
; DEBUG-NEXT:      35: 0f 85 d5 ff ff ff                jne 0x10 <foo+0x10>
; DEBUG-NEXT:      3b: 0f 1f 44 00 00                   nopl    (%rax,%rax)

; NODEBUG-NEXT:      29: 0f 85 f1 ff ff ff              jne 0x20 <foo+0x20>
; NODEBUG-NEXT:      2f: 48 8d 42 02                    leaq    2(%rdx), %rax
; NODEBUG-NEXT:      33: 0f b6 10                       movzbl  (%rax), %edx
; NODEBUG-NEXT:      36: 80 fa 5a                       cmpb    $90, %dl
; NODEBUG-NEXT:      39: 0f 85 d1 ff ff ff              jne 0x10 <foo+0x10>
; NODEBUG-NEXT:      3f: 90                             nop

; CHECK-NEXT:      40: 80 78 04 63                      cmpb    $99, 4(%rax)
; CHECK-NEXT:      44: 74 0b                            je  0x51 <foo+0x51>
; CHECK-NEXT:      46: 48 ff c0                         incq    %rax
; CHECK-NEXT:      49: e9 ba ff ff ff                   jmp 0x8 <foo+0x8>
; CHECK-NEXT:      4e: 66 90                            nop
; CHECK-NEXT:      50: c3                               retq
; CHECK-NEXT:      51: c3                               retq

; DEFAULT:            0: 48 89 f8                         movq    %rdi, %rax
; DEFAULT-NEXT:       3: 0f 1f 44 00 00                   nopl    (%rax,%rax)
; DEFAULT-NEXT:       8: 0f b6 10                         movzbl  (%rax), %edx
; DEFAULT-NEXT:       b: 80 fa 0a                         cmpb    $10, %dl
; DEFAULT-NEXT:       e: 74 30                            je  0x40 <foo+0x40>
; DEFAULT-NEXT:      10: 80 fa 64                         cmpb    $100, %dl
; DEFAULT-NEXT:      13: 74 31                            je  0x46 <foo+0x46>
; DEFAULT-NEXT:      15: 80 fa c8                         cmpb    $-56, %dl
; DEFAULT-NEXT:      18: 74 2c                            je  0x46 <foo+0x46>
; DEFAULT-NEXT:      1a: 80 fa 2c                         cmpb    $44, %dl
; DEFAULT-NEXT:      1d: 75 27                            jne 0x46 <foo+0x46>
; DEFAULT-NEXT:      1f: 90                               nop
; DEFAULT-NEXT:      20: 48 89 c2                         movq    %rax, %rdx
; DEFAULT-NEXT:      23: 48 ff c0                         incq    %rax
; DEFAULT-NEXT:      26: 80 38 1e                         cmpb    $30, (%rax)
; DEFAULT-NEXT:      29: 75 f5                            jne 0x20 <foo+0x20>
; DEFAULT-NEXT:      2b: 48 8d 42 02                      leaq    2(%rdx), %rax
; DEFAULT-NEXT:      2f: 0f b6 10                         movzbl  (%rax), %edx
; DEFAULT-NEXT:      32: 80 fa 5a                         cmpb    $90, %dl
; DEFAULT-NEXT:      35: 75 d9                            jne 0x10 <foo+0x10>
; DEFAULT-NEXT:      37: 66 0f 1f 84 00 00 00 00 00       nopw    (%rax,%rax)
; DEFAULT-NEXT:      40: 80 78 04 63                      cmpb    $99, 4(%rax)
; DEFAULT-NEXT:      44: 74 0b                            je  0x51 <foo+0x51>
; DEFAULT-NEXT:      46: 48 ff c0                         incq    %rax
; DEFAULT-NEXT:      49: eb bd                            jmp 0x8 <foo+0x8>
; DEFAULT-NEXT:      4b: 0f 1f 44 00 00                   nopl    (%rax,%rax)
; DEFAULT-NEXT:      50: c3                               retq
; DEFAULT-NEXT:      51: c3                               retq
