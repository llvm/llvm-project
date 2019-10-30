; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc-ibm-aix-xcoff \
; RUN: -code-model=small -stop-after=machine-cp < %s | FileCheck \
; RUN: --check-prefix=32SMALL-MIR %s

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc-ibm-aix-xcoff \
; RUN: -code-model=large -stop-after=machine-cp < %s | FileCheck \
; RUN: --check-prefix=32LARGE-MIR %s

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc64-ibm-aix-xcoff \
; RUN: -code-model=small -stop-after=machine-cp < %s | FileCheck \
; RUN: --check-prefix=64SMALL-MIR %s

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc64-ibm-aix-xcoff \
; RUN: -code-model=large -stop-after=machine-cp < %s | FileCheck \
; RUN: --check-prefix=64LARGE-MIR %s

  define i32 @jump_table(i32 %a) {
  entry:
    switch i32 %a, label %sw.epilog [
      i32 1, label %sw.bb
      i32 2, label %sw.bb1
      i32 3, label %sw.bb2
      i32 4, label %sw.bb3
    ]

  sw.bb:
    tail call void asm sideeffect "", ""()
    br label %sw.epilog

  sw.bb1:
    tail call void asm sideeffect "", ""()
    br label %sw.epilog

  sw.bb2:
    tail call void asm sideeffect "", ""()
    br label %sw.epilog

  sw.bb3:
    tail call void asm sideeffect "", ""()
    br label %sw.epilog

  sw.epilog:
    ret i32 0
  }


; 32SMALL-MIR: renamable $r[[REG1:[0-9]+]] = LWZtoc %jump-table.0, $r2 :: (load 4 from got)
; 32SMALL-MIR: renamable $r[[REG3:[0-9]+]] = RLWINM killed renamable $r[[REG2:[0-9]+]], 2, 0, 29
; 32SMALL-MIR: renamable $r[[REG4:[0-9]+]] = LWZX killed renamable $r[[REG3]], killed renamable $r[[REG1]] :: (load 4 from jump-table)

; 32LARGE-MIR: renamable $r[[REG1:[0-9]+]] = ADDIStocHA $r2, %jump-table.0
; 32LARGE-MIR: renamable $r[[REG2:[0-9]+]] = LWZtocL %jump-table.0, killed renamable $r[[REG1]], implicit $r2 :: (load 4 from got)
; 32LARGE-MIR: renamable $r[[REG4:[0-9]+]] = RLWINM killed renamable $r[[REG3:[0-9]+]], 2, 0, 29
; 32LARGE-MIR: renamable $r[[REG5:[0-9]+]] = LWZX killed renamable $r[[REG4]], killed renamable $r[[REG2]] :: (load 4 from jump-table)

; 64SMALL-MIR: renamable $x[[REG1:[0-9]+]] = LDtocJTI %jump-table.0, $x2 :: (load 8 from got)
; 64SMALL-MIR: renamable $x[[REG3:[0-9]+]] = RLDIC killed renamable $x[[REG2:[0-9]+]], 2, 30
; 64SMALL-MIR: renamable $x[[REG4:[0-9]+]] = LWAX killed renamable $x[[REG3]], renamable $x[[REG1]] :: (load 4 from jump-table)

; 64LARGE-MIR: renamable $x[[REG1:[0-9]+]] = ADDIStocHA8 $x2, %jump-table.0
; 64LARGE-MIR: renamable $x[[REG2:[0-9]+]] = LDtocL %jump-table.0, killed renamable $x[[REG1]], implicit $x2 :: (load 8 from got)
; 64LARGE-MIR: renamable $x[[REG4:[0-9]+]] = RLDIC killed renamable $x[[REG3:[0-9]+]], 2, 30
; 64LARGE-MIR: renamable $x[[REG5:[0-9]+]] = LWAX killed renamable $x[[REG4]], killed renamable $x[[REG2]] :: (load 4 from jump-table)
