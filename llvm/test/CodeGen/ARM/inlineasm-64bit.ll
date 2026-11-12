; RUN: llc < %s -O3  -mtriple=arm-linux-gnueabi -no-integrated-as | FileCheck %s
; RUN: llc -mtriple=thumbv7-none-linux-gnueabi -verify-machineinstrs -no-integrated-as < %s | FileCheck %s
; check if regs are passing correctly
define void @i64_write(ptr %p, i64 %val) nounwind {
; CHECK-LABEL: i64_write:
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; CHECK: strexd [[REG1]], {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}
  %1 = tail call i64 asm sideeffect "1: ldrexd $0, ${0:H}, [$2]\0A strexd $0, $3, ${3:H}, [$2]\0A teq $0, #0\0A bne 1b", "=&r,=*Qo,r,r,~{cc}"(ptr elementtype(i64) %p, ptr %p, i64 %val) nounwind
  ret void
}

; check if register allocation can reuse the registers
define void @multi_writes(ptr %p, i64 %val1, i64 %val2, i64 %val3, i64 %val4, i64 %val5, i64 %val6) nounwind {
entry:
; CHECK-LABEL: multi_writes:
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]

; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]

; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]

  tail call void asm sideeffect " strexd $1, ${1:H}, [$0]\0A strexd $2, ${2:H}, [$0]\0A strexd $3, ${3:H}, [$0]\0A strexd $4, ${4:H}, [$0]\0A strexd $5, ${5:H}, [$0]\0A strexd $6, ${6:H}, [$0]\0A", "r,r,r,r,r,r,r"(ptr %p, i64 %val1, i64 %val2, i64 %val3, i64 %val4, i64 %val5, i64 %val6) nounwind
  %incdec.ptr = getelementptr inbounds i64, ptr %p, i32 1
  tail call void asm sideeffect " strexd $1, ${1:H}, [$0]\0A strexd $2, ${2:H}, [$0]\0A strexd $3, ${3:H}, [$0]\0A strexd $4, ${4:H}, [$0]\0A strexd $5, ${5:H}, [$0]\0A strexd $6, ${6:H}, [$0]\0A", "r,r,r,r,r,r,r"(ptr %incdec.ptr, i64 %val1, i64 %val2, i64 %val3, i64 %val4, i64 %val5, i64 %val6) nounwind
  tail call void asm sideeffect " strexd $1, ${1:H}, [$0]\0A strexd $2, ${2:H}, [$0]\0A strexd $3, ${3:H}, [$0]\0A strexd $4, ${4:H}, [$0]\0A strexd $5, ${5:H}, [$0]\0A strexd $6, ${6:H}, [$0]\0A", "r,r,r,r,r,r,r"(ptr %incdec.ptr, i64 %val1, i64 %val2, i64 %val3, i64 %val4, i64 %val5, i64 %val6) nounwind
  ret void
}


; check if callee-saved registers used by inline asm are saved/restored
define void @foo(ptr %p, i64 %i) nounwind {
; CHECK-LABEL:foo:
; CHECK: {{push|push.w}} {{{r[4-9]|r10|r11}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; CHECK: strexd [[REG1]], {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}
; CHECK: {{pop|pop.w}} {{{r[4-9]|r10|r11}}
  %1 = tail call { i64, i64 } asm sideeffect "@ atomic64_set\0A1: ldrexd $0, ${0:H}, [$3]\0Aldrexd $1, ${1:H}, [$3]\0A strexd $0, $4, ${4:H}, [$3]\0A teq $0, #0\0A bne 1b", "=&r,=&r,=*Qo,r,r,~{cc}"(ptr elementtype(i64) %p, ptr %p, i64 %i) nounwind
  ret void
}

; return *p;
define i64 @ldrd_test(ptr %p) nounwind {
; CHECK-LABEL: ldrd_test:
  %1 = tail call i64 asm "ldrd $0, ${0:H}, [$1]", "=r,r"(ptr %p) nounwind
  ret i64 %1
}

define i64 @QR_test(ptr %p) nounwind {
; CHECK-LABEL: QR_test:
; CHECK: ldrd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}
  %1 = tail call i64 asm "ldrd ${0:Q}, ${0:R}, [$1]", "=r,r"(ptr %p) nounwind
  ret i64 %1
}

define i64 @defuse_test(i64 %p) nounwind {
; CHECK-LABEL: defuse_test:
; CHECK: add {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, #1
  %1 = tail call i64 asm "add $0, ${0:H}, #1", "=r,0"(i64 %p) nounwind
  ret i64 %1
}

; *p = (hi << 32) | lo;
define void @strd_test(ptr %p, i32 %lo, i32 %hi) nounwind {
; CHECK-LABEL: strd_test:
; CHECK: strd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}
  %1 = zext i32 %hi to i64
  %2 = shl nuw i64 %1, 32
  %3 = sext i32 %lo to i64
  %4 = or i64 %2, %3
  tail call void asm sideeffect "strd $0, ${0:H}, [$1]", "r,r"(i64 %4, ptr %p) nounwind
  ret void
}

; Make sure we don't untie operands by mistake.
define i64 @tied_64bit_test(i64 %in) nounwind {
; CHECK-LABEL: tied_64bit_test:
; CHECK: OUT([[OUTREG:r[0-9]+]]), IN([[OUTREG]])
  %addr = alloca i64
  call void asm "OUT($0), IN($1)", "=*rm,0"(ptr elementtype(i64) %addr, i64 %in)
  ret i64 %in
}

; If we explicitly name a tied operand, then the code should lookup the operand
; we were tied to for information about register class and so on.
define i64 @tied_64bit_lookback_test(i64 %in) nounwind {
; CHECK-LABEL: tied_64bit_lookback_test:
; CHECK: OUTLO([[LO:r[0-9]+]]) OUTHI([[HI:r[0-9]+]]) INLO([[LO]]) INHI([[HI]])
  %vars = call {i64, i32, i64} asm "OUTLO(${2:Q}) OUTHI(${2:R}) INLO(${3:Q}) INHI(${3:R})", "=r,=r,=r,2"(i64 %in)
  %res = extractvalue {i64, i32, i64} %vars, 2
  ret i64 %res
}

; Check access to low and high part with a specific register pair constraint
define i64 @low_high_specific_reg_pair(i64 %in) nounwind {
; CHECK-LABEL: low_high_specific_reg_pair
; CHECK: mov r3, r2
  %res = call i64 asm "mov ${0:R}, ${1:Q}", "=&{r2},0"(i64 %in)
  ret i64 %res
}
