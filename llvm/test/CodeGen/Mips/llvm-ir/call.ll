; Test the 'call' instruction and the tailcall variant.

; RUN: llc -march=mips   -mcpu=mips32   -relocation-model=pic  -mips-tail-calls=1 < %s | FileCheck %s -check-prefixes=ALL,O32,NOT-R6C
; RUN: llc -march=mips   -mcpu=mips32r2 -relocation-model=pic  -mips-tail-calls=1 < %s | FileCheck %s -check-prefixes=ALL,O32,NOT-R6C
; RUN: llc -march=mips   -mcpu=mips32r3 -relocation-model=pic  -mips-tail-calls=1 < %s | FileCheck %s -check-prefixes=ALL,O32,NOT-R6C
; RUN: llc -march=mips   -mcpu=mips32r5 -relocation-model=pic  -mips-tail-calls=1 < %s | FileCheck %s -check-prefixes=ALL,O32,NOT-R6C
; RUN: llc -march=mips   -mcpu=mips32r6 -relocation-model=pic -disable-mips-delay-filler  -mips-tail-calls=1 < %s | FileCheck %s -check-prefixes=ALL,O32,R6C
; RUN: llc -march=mips   -mcpu=mips32r6 -relocation-model=pic -mattr=+fp64,+nooddspreg -disable-mips-delay-filler  -mips-tail-calls=1 < %s | FileCheck %s -check-prefixes=ALL,O32,R6C
; RUN: llc -march=mips64 -mcpu=mips4    -relocation-model=pic   -mips-tail-calls=1 < %s | FileCheck %s -check-prefixes=ALL,N64,NOT-R6C
; RUN: llc -march=mips64 -mcpu=mips64   -relocation-model=pic   -mips-tail-calls=1 < %s | FileCheck %s -check-prefixes=ALL,N64,NOT-R6C
; RUN: llc -march=mips64 -mcpu=mips64r2 -relocation-model=pic   -mips-tail-calls=1 < %s | FileCheck %s -check-prefixes=ALL,N64,NOT-R6C
; RUN: llc -march=mips64 -mcpu=mips64r3 -relocation-model=pic   -mips-tail-calls=1 < %s | FileCheck %s -check-prefixes=ALL,N64,NOT-R6C
; RUN: llc -march=mips64 -mcpu=mips64r5 -relocation-model=pic   -mips-tail-calls=1 < %s | FileCheck %s -check-prefixes=ALL,N64,NOT-R6C
; RUN: llc -march=mips64 -mcpu=mips64r6 -relocation-model=pic  -disable-mips-delay-filler  -mips-tail-calls=1 < %s | FileCheck %s -check-prefixes=ALL,N64,R6C
; RUN: llc -march=mips   -mcpu=mips32   -relocation-model=pic  -mips-tail-calls=1 < %s | FileCheck %s -check-prefix=ALL -check-prefix=O32 -check-prefix=NOT-R6C
; RUN: llc -march=mips   -mcpu=mips32r2 -relocation-model=pic  -mips-tail-calls=1 < %s | FileCheck %s -check-prefix=ALL -check-prefix=O32 -check-prefix=NOT-R6C
; RUN: llc -march=mips   -mcpu=mips32r3 -relocation-model=pic  -mips-tail-calls=1 < %s | FileCheck %s -check-prefix=ALL -check-prefix=O32 -check-prefix=NOT-R6C
; RUN: llc -march=mips   -mcpu=mips32r5 -relocation-model=pic  -mips-tail-calls=1 < %s | FileCheck %s -check-prefix=ALL -check-prefix=O32 -check-prefix=NOT-R6C
; RUN: llc -march=mips   -mcpu=mips32r6 -relocation-model=pic -disable-mips-delay-filler  -mips-tail-calls=1 < %s | FileCheck %s -check-prefix=ALL -check-prefix=O32 -check-prefix=R6C
; RUN: llc -march=mips   -mcpu=mips32r6 -relocation-model=pic -mattr=+fp64,+nooddspreg -disable-mips-delay-filler  -mips-tail-calls=1 < %s | FileCheck %s -check-prefix=ALL -check-prefix=O32 -check-prefix=R6C
; RUN: llc -march=mips64 -mcpu=mips4    -relocation-model=pic   -mips-tail-calls=1 < %s | FileCheck %s -check-prefix=ALL -check-prefix=N64 -check-prefix=NOT-R6C
; RUN: llc -march=mips64 -mcpu=mips64   -relocation-model=pic   -mips-tail-calls=1 < %s | FileCheck %s -check-prefix=ALL -check-prefix=N64 -check-prefix=NOT-R6C
; RUN: llc -march=mips64 -mcpu=mips64r2 -relocation-model=pic   -mips-tail-calls=1 < %s | FileCheck %s -check-prefix=ALL -check-prefix=N64 -check-prefix=NOT-R6C
; RUN: llc -march=mips64 -mcpu=mips64r3 -relocation-model=pic   -mips-tail-calls=1 < %s | FileCheck %s -check-prefix=ALL -check-prefix=N64 -check-prefix=NOT-R6C
; RUN: llc -march=mips64 -mcpu=mips64r5 -relocation-model=pic   -mips-tail-calls=1 < %s | FileCheck %s -check-prefix=ALL -check-prefix=N64 -check-prefix=NOT-R6C
; RUN: llc -march=mips64 -mcpu=mips64r6 -relocation-model=pic  -disable-mips-delay-filler  -mips-tail-calls=1 < %s | FileCheck %s -check-prefix=ALL -check-prefix=N64 -check-prefix=R6C

declare void @extern_void_void()
declare i32 @extern_i32_void()
declare float @extern_float_void()

define i32 @call_void_void() {
; ALL-LABEL: call_void_void:

; O32:           lw $[[TGT:[0-9]+]], %call16(extern_void_void)($gp)

; N64:           ld $[[TGT:[0-9]+]], %call16(extern_void_void)($gp)

; NOT-R6C:       jalr $[[TGT]]
; R6C:           jalrc $[[TGT]]

  call void @extern_void_void()
; R6C:           jrc $ra
  ret i32 0
}

define i32 @call_i32_void() {
; ALL-LABEL: call_i32_void:

; O32:           lw $[[TGT:[0-9]+]], %call16(extern_i32_void)($gp)

; N64:           ld $[[TGT:[0-9]+]], %call16(extern_i32_void)($gp)

; NOT-R6C:       jalr $[[TGT]]
; R6C:           jalrc $[[TGT]]

  %1 = call i32 @extern_i32_void()
  %2 = add i32 %1, 1
; R6C:           jrc $ra
  ret i32 %2
}

define float @call_float_void() {
; ALL-LABEL: call_float_void:

; FIXME: Not sure why we don't use $gp directly on such a simple test. We should
;        look into it at some point.
; O32:           addu $[[GP:[0-9]+]], ${{[0-9]+}}, $25
; O32:           lw $[[TGT:[0-9]+]], %call16(extern_float_void)($[[GP]])

; N64:           ld $[[TGT:[0-9]+]], %call16(extern_float_void)($gp)

; NOT-R6C:       jalr $[[TGT]]
; R6C:           jalrc $[[TGT]]


  %1 = call float @extern_float_void()
  %2 = fadd float %1, 1.0
; R6C:           jrc $ra
  ret float %2
}

define i32 @indirect_call_void_void(void ()* %addr) {
; ALL-LABEL: indirect_call_void_void:

; ALL:           move $25, $4
; NOT-R6C:       jalr $25
; R6C:           jalrc $25

  call void %addr()
; R6C:           jrc $ra
  ret i32 0
}

define i32 @indirect_call_i32_void(i32 ()* %addr) {
; ALL-LABEL: indirect_call_i32_void:

; ALL:           move $25, $4
; NOT-R6C:       jalr $25
; R6C:           jalrc $25


  %1 = call i32 %addr()
  %2 = add i32 %1, 1
; R6C:           jrc $ra
  ret i32 %2
}

define float @indirect_call_float_void(float ()* %addr) {
; ALL-LABEL: indirect_call_float_void:

; ALL:           move $25, $4
; NOT-R6C:       jalr $25
; R6C:           jalrc $25


  %1 = call float %addr()
  %2 = fadd float %1, 1.0
; R6C:           jrc $ra
  ret float %2
}

; We can't use 'musttail' here because the verifier is too conservative and
; prohibits any prototype difference.
define void @tail_indirect_call_void_void(void ()* %addr) {
; ALL-LABEL: tail_indirect_call_void_void:

; ALL:           move $25, $4
; NOT-R6C:       jr   $[[TGT]]
; R6C:           jrc  $[[TGT]]

  tail call void %addr()
  ret void
}

define i32 @tail_indirect_call_i32_void(i32 ()* %addr) {
; ALL-LABEL: tail_indirect_call_i32_void:

; ALL:           move $25, $4
; NOT-R6C:       jr   $[[TGT]]
; R6C:           jrc  $[[TGT]]

  %1 = tail call i32 %addr()
  ret i32 %1
}

define float @tail_indirect_call_float_void(float ()* %addr) {
; ALL-LABEL: tail_indirect_call_float_void:

; ALL:           move $25, $4
; NOT-R6C:       jr   $[[TGT]]
; R6C:           jrc  $[[TGT]]

  %1 = tail call float %addr()
  ret float %1
}

; Check that passing undef as a double value doesn't cause machine code errors
; for FP64.
declare hidden void @undef_double(i32 %this, double %volume) unnamed_addr align 2

define hidden void @thunk_undef_double(i32 %this, double %volume) unnamed_addr align 2 {
; ALL-LABEL: thunk_undef_double:
; O32: # implicit-def: %a2
; O32: # implicit-def: %a3
; NOT-R6C:    jr   $[[TGT]]
; R6C:        jrc  $[[TGT]]

  tail call void @undef_double(i32 undef, double undef) #8
  ret void
}

; Check that immediate addresses do not use jal.
define i32 @jal_only_allows_symbols() {
; ALL-LABEL: jal_only_allows_symbols:

; ALL-NOT:       {{jal }}
; ALL:           addiu $[[TGT:[0-9]+]], $zero, 1234
; ALL-NOT:       {{jal }}
; NOT-R6C:       jalr $[[TGT]]
; R6C:           jalrc $[[TGT]]
; ALL-NOT:       {{jal }}

  call void () inttoptr (i32 1234 to void ()*)()
; R6C:           jrc $ra
  ret i32 0
}

