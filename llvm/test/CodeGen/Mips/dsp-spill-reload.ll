; RUN: llc -march=mips -mattr=+dsp < %s -asm-show-inst -O0 | FileCheck %s \
; RUN:   --check-prefixes=ASM,ALL
; RUN: llc -march=mips -mattr=+dsp,+micromips < %s -O0 -filetype=obj | \
; RUN:   llvm-objdump --no-print-imm-hex -d - | FileCheck %s --check-prefixes=MM-OBJ,ALL

; Test that spill and reloads use the dsp "variant" instructions. We use -O0
; to use the simple register allocator.

; To test the micromips output, we have to take a round trip through the
; object file encoder/decoder as the instruction mapping tables are used to
; support micromips.

; FIXME: We should be able to get rid of those instructions with the variable
;        value registers.

; ALL-LABEL: spill_reload{{>?}}:

define <4 x i8>  @spill_reload(<4 x i8> %a, <4 x i8> %b, i32 %g) {
entry:
  %c = tail call <4 x i8> @llvm.mips.addu.qb(<4 x i8> %a, <4 x i8> %b)
  %cond = icmp eq i32 %g, 0
  br i1 %cond, label %true, label %end

; ASM: SWDSP
; ASM: SWDSP

; MM-OBJ:   sw  ${{[0-9]+}}, {{[0-9]+}}($sp)
; MM-OBJ:   sw  ${{[0-9]+}}, {{[0-9]+}}($sp)

true:
  ret <4 x i8> %c

; ASM: LWDSP

; MM-OBJ: lw ${{[0-9]+}}, {{[0-9]+}}($sp)

end:
  %d = tail call <4 x i8> @llvm.mips.addu.qb(<4 x i8> %c, <4 x i8> %a)
  ret <4 x i8> %d

; ASM: LWDSP
; ASM: LWDSP

; MM-OBJ: lw ${{[0-9]+}}, {{[0-9]+}}($sp)
; MM-OBJ: lw ${{[0-9]+}}, {{[0-9]+}}($sp)

}

declare <4 x i8> @llvm.mips.addu.qb(<4 x i8>, <4 x i8>) nounwind
