; RUN: llc < %s -march=mips -mcpu=mips2 -relocation-model=pic | FileCheck %s \
; RUN:    --check-prefixes=ALL,GP32,GP32-M
; RUN: llc < %s -march=mips -mcpu=mips32 -relocation-model=pic | FileCheck %s \
; RUN:    --check-prefixes=ALL,GP32,GP32-M
; RUN: llc < %s -march=mips -mcpu=mips32r6 -relocation-model=pic | FileCheck %s \
; RUN:    --check-prefixes=ALL,GP32,GP32-M
; RUN: llc < %s -march=mips -mcpu=mips32r2 -mattr=+micromips -relocation-model=pic | FileCheck %s \
; RUN:    --check-prefixes=ALL,GP32,GP32-MM,GP32-MMR2
; RUN: llc < %s -march=mips -mcpu=mips32r6 -mattr=+micromips -relocation-model=pic | FileCheck %s \
; RUN:    --check-prefixes=ALL,GP32,GP32-MM,GP32-MMR6
; RUN: llc < %s -march=mips64 -mcpu=mips3 -relocation-model=pic | FileCheck %s \
; RUN:    --check-prefixes=ALL,GP64,N64
; RUN: llc < %s -march=mips64 -mcpu=mips64 -relocation-model=pic | FileCheck %s \
; RUN:    --check-prefixes=ALL,GP64,N64
; RUN: llc < %s -march=mips64 -mcpu=mips64r6 -relocation-model=pic | FileCheck %s \
; RUN:    --check-prefixes=ALL,GP64,N64
; RUN: llc < %s -march=mips64 -mcpu=mips3 -target-abi n32 -relocation-model=pic | FileCheck %s \
; RUN:    --check-prefixes=ALL,GP64,N32
; RUN: llc < %s -march=mips64 -mcpu=mips64 -target-abi n32 -relocation-model=pic | FileCheck %s \
; RUN:    --check-prefixes=ALL,GP64,N32
; RUN: llc < %s -march=mips64 -mcpu=mips64r6 -target-abi n32 -relocation-model=pic | FileCheck %s \
; RUN:    --check-prefixes=ALL,GP64,N32

; Check dynamic stack realignment in functions without variable-sized objects.

declare void @helper_01(i32, i32, i32, i32, i32*)

; O32 ABI
define void @func_01() {
entry:
; GP32-LABEL: func_01:

  ; prologue
  ; FIXME: We are currently over-allocating stack space. This particular case
  ;        needs a frame of up to between 16 and 512-bytes but currently
  ;        allocates between 1024 and 1536 bytes
  ; GP32-M:     addiu   $sp, $sp, -1024
  ; GP32-MMR2:  addiusp -1024
  ; GP32-MMR6:  addiu   $sp, $sp, -1024
  ; GP32:       sw      $ra, 1020($sp)
  ; GP32:       sw      $fp, 1016($sp)
  ;
  ; GP32:       move    $fp, $sp
  ; GP32:       addiu   $[[T0:[0-9]+|ra|gp]], $zero, -512
  ; GP32-NEXT:  and     $sp, $sp, $[[T0]]

  ; body
  ; GP32:       addiu   $[[T1:[0-9]+]], $sp, 512
  ; GP32-M:     sw      $[[T1]], 16($sp)
  ; GP32-MM:    sw16    $[[T1]], 16(${{[0-9]+}})

  ; epilogue
  ; GP32:       move    $sp, $fp
  ; GP32:       lw      $fp, 1016($sp)
  ; GP32:       lw      $ra, 1020($sp)
  ; GP32-M:     addiu   $sp, $sp, 1024
  ; GP32-MMR2:  addiusp 1024
  ; GP32-MMR6:  addiu   $sp, $sp, 1024

  %a = alloca i32, align 512
  call void @helper_01(i32 0, i32 0, i32 0, i32 0, i32* %a)
  ret void
}

declare void @helper_02(i32, i32, i32, i32,
                        i32, i32, i32, i32, i32*)

; N32/N64 ABIs
define void @func_02() {
entry:
; GP64-LABEL: func_02:

  ; prologue
  ; FIXME: We are currently over-allocating stack space. This particular case
  ;        needs a frame of up to between 16 and 512-bytes but currently
  ;        allocates between 1024 and 1536 bytes
  ; N32:        addiu   $sp, $sp, -1024
  ; N64:        daddiu  $sp, $sp, -1024
  ; GP64:       sd      $ra, 1016($sp)
  ; GP64:       sd      $fp, 1008($sp)
  ; N32:        sd      $gp, 1000($sp)
  ;
  ; GP64:       move    $fp, $sp
  ; N32:        addiu   $[[T0:[0-9]+|ra]], $zero, -512
  ; N64:        daddiu  $[[T0:[0-9]+|ra]], $zero, -512
  ; GP64-NEXT:  and     $sp, $sp, $[[T0]]

  ; body
  ; N32:        addiu   $[[T1:[0-9]+]], $sp, 512
  ; N64:        daddiu  $[[T1:[0-9]+]], $sp, 512
  ; GP64:       sd      $[[T1]], 0($sp)

  ; epilogue
  ; GP64:       move    $sp, $fp
  ; N32:        ld      $gp, 1000($sp)
  ; GP64:       ld      $fp, 1008($sp)
  ; GP64:       ld      $ra, 1016($sp)
  ; N32:        addiu   $sp, $sp, 1024
  ; N64:        daddiu  $sp, $sp, 1024

  %a = alloca i32, align 512
  call void @helper_02(i32 0, i32 0, i32 0, i32 0,
                       i32 0, i32 0, i32 0, i32 0, i32* %a)
  ret void
}

; Verify that we use $fp for referencing incoming arguments.

declare void @helper_03(i32, i32, i32, i32, i32*, i32*)

; O32 ABI
define void @func_03(i32 %p0, i32 %p1, i32 %p2, i32 %p3, i32* %b) {
entry:
; GP32-LABEL: func_03:

  ; body
  ; FIXME: We are currently over-allocating stack space.
  ; GP32-DAG:     addiu   $[[T0:[0-9]+]], $sp, 512
  ; GP32-M-DAG:   sw      $[[T0]], 16($sp)
  ; GP32-MM-DAG:  sw16    $[[T0]], 16(${{[0-9]+}})
  ; GP32-DAG:     lw      $[[T1:[0-9]+]], 1040($fp)
  ; GP32-M-DAG:   sw      $[[T1]], 20($sp)
  ; GP32-MM-DAG:  sw16    $[[T1]], 20(${{[0-9]+}})

  %a = alloca i32, align 512
  call void @helper_03(i32 0, i32 0, i32 0, i32 0, i32* %a, i32* %b)
  ret void
}

declare void @helper_04(i32, i32, i32, i32,
                        i32, i32, i32, i32, i32*, i32*)

; N32/N64 ABIs
define void @func_04(i32 %p0, i32 %p1, i32 %p2, i32 %p3,
                     i32 %p4, i32 %p5, i32 %p6, i32 %p7,
                     i32* %b) {
entry:
; GP64-LABEL: func_04:

  ; body
  ; FIXME: We are currently over-allocating stack space.
  ; N32-DAG:    addiu   $[[T0:[0-9]+]], $sp, 512
  ; N64-DAG:    daddiu  $[[T0:[0-9]+]], $sp, 512
  ; GP64-DAG:   sd      $[[T0]], 0($sp)
  ; GP64-DAG:   ld      $[[T1:[0-9]+]], 1024($fp)
  ; GP64-DAG:   sd      $[[T1]], 8($sp)

  %a = alloca i32, align 512
  call void @helper_04(i32 0, i32 0, i32 0, i32 0,
                       i32 0, i32 0, i32 0, i32 0, i32* %a, i32* %b)
  ret void
}

; Check dynamic stack realignment in functions with variable-sized objects.

; O32 ABI
define void @func_05(i32 %sz) {
entry:
; GP32-LABEL: func_05:

  ; prologue
  ; FIXME: We are currently over-allocating stack space.
  ; GP32-M:     addiu   $sp, $sp, -1024
  ; GP32-MMR2:  addiusp -1024
  ; GP32-MMR6:  addiu   $sp, $sp, -1024
  ; GP32:       sw      $fp, 1020($sp)
  ; GP32:       sw      $23, 1016($sp)
  ;
  ; GP32:       move    $fp, $sp
  ; GP32:       addiu   $[[T0:[0-9]+|gp]], $zero, -512
  ; GP32-NEXT:  and     $sp, $sp, $[[T0]]
  ; GP32-NEXT:  move    $23, $sp

  ; body
  ; GP32:       addiu   $[[T1:[0-9]+]], $zero, 222
  ; GP32:       sw      $[[T1]], 508($23)

  ; epilogue
  ; GP32:       move    $sp, $fp
  ; GP32:       lw      $23, 1016($sp)
  ; GP32:       lw      $fp, 1020($sp)
  ; GP32-M:     addiu   $sp, $sp, 1024
  ; GP32-MMR2:  addiusp 1024
  ; GP32-MMR6:  addiu   $sp, $sp, 1024

  %a0 = alloca i32, i32 %sz, align 512
  %a1 = alloca i32, align 4

  store volatile i32 111, i32* %a0, align 512
  store volatile i32 222, i32* %a1, align 4

  ret void
}

; N32/N64 ABIs
define void @func_06(i32 %sz) {
entry:
; GP64-LABEL: func_06:

  ; prologue
  ; FIXME: We are currently over-allocating stack space.
  ; N32:        addiu   $sp, $sp, -1024
  ; N64:        daddiu  $sp, $sp, -1024
  ; GP64:       sd      $fp, 1016($sp)
  ; GP64:       sd      $23, 1008($sp)
  ;
  ; GP64:       move    $fp, $sp
  ; GP64:       addiu   $[[T0:[0-9]+|gp]], $zero, -512
  ; GP64-NEXT:  and     $sp, $sp, $[[T0]]
  ; GP64-NEXT:  move    $23, $sp

  ; body
  ; GP64:       addiu   $[[T1:[0-9]+]], $zero, 222
  ; GP64:       sw      $[[T1]], 508($23)

  ; epilogue
  ; GP64:       move    $sp, $fp
  ; GP64:       ld      $23, 1008($sp)
  ; GP64:       ld      $fp, 1016($sp)
  ; N32:        addiu   $sp, $sp, 1024
  ; N64:        daddiu  $sp, $sp, 1024

  %a0 = alloca i32, i32 %sz, align 512
  %a1 = alloca i32, align 4

  store volatile i32 111, i32* %a0, align 512
  store volatile i32 222, i32* %a1, align 4

  ret void
}

; Verify that we use $fp for referencing incoming arguments and $sp for
; building outbound arguments for nested function calls.

; O32 ABI
define void @func_07(i32 %p0, i32 %p1, i32 %p2, i32 %p3, i32 %sz) {
entry:
; GP32-LABEL: func_07:

  ; body
  ; FIXME: We are currently over-allocating stack space.
  ; GP32-DAG:       lw      $[[T0:[0-9]+]], 1040($fp)
  ;
  ; GP32-DAG:       addiu   $[[T1:[0-9]+]], $zero, 222
  ; GP32-DAG:       sw      $[[T1]], 508($23)
  ;
  ; GP32-M-DAG:     sw      $[[T2:[0-9]+]], 16($sp)
  ; GP32-MM-DAG:    sw16    $[[T2:[0-9]+]], 16($[[T3:[0-9]+]])

  %a0 = alloca i32, i32 %sz, align 512
  %a1 = alloca i32, align 4

  store volatile i32 111, i32* %a0, align 512
  store volatile i32 222, i32* %a1, align 4

  call void @helper_01(i32 0, i32 0, i32 0, i32 0, i32* %a1)

  ret void
}

; N32/N64 ABIs
define void @func_08(i32 %p0, i32 %p1, i32 %p2, i32 %p3,
                     i32 %p4, i32 %p5, i32 %p6, i32 %p7,
                     i32 %sz) {
entry:
; GP64-LABEL: func_08:

  ; body
  ; FIXME: We are currently over-allocating stack space.
  ; N32-DAG:        lw      $[[T0:[0-9]+]], 1028($fp)
  ; N64-DAG:        lwu     $[[T0:[0-9]+]], 1028($fp)
  ;
  ; GP64-DAG:       addiu   $[[T1:[0-9]+]], $zero, 222
  ; GP64-DAG:       sw      $[[T1]], 508($23)
  ;
  ; GP64-DAG:       sd      $[[T2:[0-9]+]], 0($sp)

  %a0 = alloca i32, i32 %sz, align 512
  %a1 = alloca i32, align 4

  store volatile i32 111, i32* %a0, align 512
  store volatile i32 222, i32* %a1, align 4

  call void @helper_02(i32 0, i32 0, i32 0, i32 0,
                       i32 0, i32 0, i32 0, i32 0, i32* %a1)
  ret void
}

; Check that we do not perform dynamic stack realignment in the presence of
; the "no-realign-stack" function attribute.
define void @func_09() "no-realign-stack" {
entry:
; ALL-LABEL: func_09:

  ; ALL-NOT:  and     $sp, $sp, $[[T0:[0-9]+|ra|gp]]

  %a = alloca i32, align 512
  call void @helper_01(i32 0, i32 0, i32 0, i32 0, i32* %a)
  ret void
}

define void @func_10(i32 %sz) "no-realign-stack" {
entry:
; ALL-LABEL: func_10:

  ; ALL-NOT:  and     $sp, $sp, $[[T0:[0-9]+|ra|gp]]

  %a0 = alloca i32, i32 %sz, align 512
  %a1 = alloca i32, align 4

  store volatile i32 111, i32* %a0, align 512
  store volatile i32 222, i32* %a1, align 4

  ret void
}
