; RUN: llc -mtriple=thumbv7-linux-gnueabihf -o - -show-mc-encoding -t2-reduce-limit=0 -t2-reduce-limit2=0 %s | FileCheck %s
; RUN: llc -mtriple=thumbv7-linux-gnueabihf -o - -show-mc-encoding %s | FileCheck %s --check-prefix=CHECK-OPT

define i32 @and(i32 %a, i32 %b) nounwind readnone {
; CHECK-LABEL: and:
; CHECK: and.w r{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}} @ encoding: [{{0x..,0x..,0x..,0x..}}]
; CHECK-OPT: ands r{{[0-7]}}, r{{[0-7]}} @ encoding: [{{0x..,0x..}}]
entry:
  %and = and i32 %b, %a
  ret i32 %and
}

define i32 @asr-imm(i32 %a) nounwind readnone {
; CHECK-LABEL: "asr-imm":
; CHECK: asr.w r{{[0-9]+}}, r{{[0-9]+}}, #13 @ encoding: [{{0x..,0x..,0x..,0x..}}]
; CHECK-OPT: asrs r{{[0-7]}}, r{{[0-7]}}, #13 @ encoding: [{{0x..,0x..}}]
entry:
  %shr = ashr i32 %a, 13
  ret i32 %shr
}

define i32 @asr-reg(i32 %a, i32 %b) nounwind readnone {
; CHECK-LABEL: "asr-reg":
; CHECK: asr.w r{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}} @ encoding: [{{0x..,0x..,0x..,0x..}}]
; CHECK-OPT: asrs r{{[0-7]}}, r{{[0-7]}} @ encoding: [{{0x..,0x..}}]
entry:
  %shr = ashr i32 %a, %b
  ret i32 %shr
}

define i32 @bic(i32 %a, i32 %b) nounwind readnone {
; CHECK-LABEL: bic:
; CHECK: bic.w r{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}} @ encoding: [{{0x..,0x..,0x..,0x..}}]
; CHECK-OPT: bics r{{[0-7]}}, r{{[0-7]}} @ encoding: [{{0x..,0x..}}]
entry:
  %neg = xor i32 %b, -1
  %and = and i32 %neg, %a
  ret i32 %and
}

define i32 @eor(i32 %a, i32 %b) nounwind readnone {
; CHECK-LABEL: eor:
; CHECK: eor.w r{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}} @ encoding: [{{0x..,0x..,0x..,0x..}}]
; CHECK-OPT: eors r{{[0-7]}}, r{{[0-7]}} @ encoding: [{{0x..,0x..}}]
entry:
  %eor = xor i32 %a, %b
  ret i32 %eor
}

define i32 @lsl-imm(i32 %a) nounwind readnone {
; CHECK-LABEL: "lsl-imm":
; CHECK: lsl.w r{{[0-9]+}}, r{{[0-9]+}}, #13 @ encoding: [{{0x..,0x..,0x..,0x..}}]
; CHECK-OPT: lsls r{{[0-7]}}, r{{[0-7]}}, #13  @ encoding: [{{0x..,0x..}}]
entry:
  %shl = shl i32 %a, 13
  ret i32 %shl
}

define i32 @lsl-reg(i32 %a, i32 %b) nounwind readnone {
; CHECK-LABEL: "lsl-reg":
; CHECK: lsl.w r{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}} @ encoding: [{{0x..,0x..,0x..,0x..}}]
; CHECK-OPT: lsls r{{[0-7]}}, r{{[0-7]}}  @ encoding: [{{0x..,0x..}}]
entry:
  %shl = shl i32 %a, %b
  ret i32 %shl
}

define i32 @lsr-imm(i32 %a) nounwind readnone {
; CHECK-LABEL: "lsr-imm":
; CHECK: lsr.w r{{[0-9]+}}, r{{[0-9]+}}, #13 @ encoding: [{{0x..,0x..,0x..,0x..}}]
; CHECK-OPT: lsrs r{{[0-7]}}, r{{[0-7]}}, #13  @ encoding: [{{0x..,0x..}}]
entry:
  %shr = lshr i32 %a, 13
  ret i32 %shr
}

define i32 @lsr-reg(i32 %a, i32 %b) nounwind readnone {
; CHECK-LABEL: "lsr-reg":
; CHECK: lsr.w r{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}} @ encoding: [{{0x..,0x..,0x..,0x..}}]
; CHECK-OPT: lsrs r{{[0-7]}}, r{{[0-7]}}  @ encoding: [{{0x..,0x..}}]
entry:
  %shr = lshr i32 %a, %b
  ret i32 %shr
}

define i32 @bundled_instruction(ptr %addr, ptr %addr2, i1 %tst) minsize {
; CHECK-LABEL: bundled_instruction:
; CHECK: itee ne
; CHECK: ldmeq r0!, {{{r[0-9]+}}}
  br i1 %tst, label %true, label %false

true:
  ret i32 0

false:
  %res = load i32, ptr %addr, align 4
  %next = getelementptr i32, ptr %addr, i32 1
  store ptr %next, ptr %addr2
  ret i32 %res
}

; ldm instructions fault on misaligned accesses so we mustn't convert
; this post-indexed ldr into one.
define ptr @misaligned_post(ptr %src, ptr %dest) minsize {
; CHECK-LABEL: misaligned_post:
; CHECK: ldr [[VAL:.*]], [r0], #4
; CHECK: str [[VAL]], [r1]

  %val = load i32, ptr %src, align 1
  store i32 %val, ptr %dest
  %next = getelementptr i32, ptr %src, i32 1
  ret ptr %next
}
