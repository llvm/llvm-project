; RUN: llc -verify-machineinstrs -mcpu=pwr10 -mtriple=powerpc64le-unknown-unknown \
; RUN:   -ppc-asm-full-reg-names --ppc-vsr-nums-as-vr < %s | FileCheck %s

; Function Attrs: nounwind
; CHECK-LABEL: and_not
; CHECK:         xxlandc v2, v2, v3
; CHECK-NEXT:    blr
define dso_local <4 x i32> @and_not(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C) local_unnamed_addr #0 {
entry:
  %neg = xor <4 x i32> %B, <i32 -1, i32 -1, i32 -1, i32 -1>
  %and = and <4 x i32> %neg, %A
  ret <4 x i32> %and
}

; Function Attrs: nounwind
; CHECK-LABEL: and_and8
; CHECK:         xxeval v2, v3, v2, v4, 1
; CHECK-NEXT:    blr
define dso_local <16 x i8> @and_and8(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C) local_unnamed_addr #0 {
entry:
  %and = and <16 x i8> %B, %A
  %and1 = and <16 x i8> %and, %C
  ret <16 x i8> %and1
}

; Function Attrs: nounwind
; CHECK-LABEL: and_and16
; CHECK:         xxeval v2, v3, v2, v4, 1
; CHECK-NEXT:    blr
define dso_local <8 x i16> @and_and16(<8 x i16> %A, <8 x i16> %B, <8 x i16> %C) local_unnamed_addr #0 {
entry:
  %and = and <8 x i16> %B, %A
  %and1 = and <8 x i16> %and, %C
  ret <8 x i16> %and1
}

; Function Attrs: nounwind
; CHECK-LABEL: and_and32
; CHECK:         xxeval v2, v3, v2, v4, 1
; CHECK-NEXT:    blr
define dso_local <4 x i32> @and_and32(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C) local_unnamed_addr #0 {
entry:
  %and = and <4 x i32> %B, %A
  %and1 = and <4 x i32> %and, %C
  ret <4 x i32> %and1
}

; Function Attrs: nounwind
; CHECK-LABEL: and_and64
; CHECK:         xxeval v2, v3, v2, v4, 1
; CHECK-NEXT:    blr
define dso_local <2 x i64> @and_and64(<2 x i64> %A, <2 x i64> %B, <2 x i64> %C) local_unnamed_addr #0 {
entry:
  %and = and <2 x i64> %B, %A
  %and1 = and <2 x i64> %and, %C
  ret <2 x i64> %and1
}

; Function Attrs: nounwind
; CHECK-LABEL: and_nand
; CHECK:         xxeval v2, v2, v4, v3, 14
; CHECK-NEXT:    blr
define dso_local <4 x i32> @and_nand(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C) local_unnamed_addr #0 {
entry:
  %and = and <4 x i32> %C, %B
  %neg = xor <4 x i32> %and, <i32 -1, i32 -1, i32 -1, i32 -1>
  %and1 = and <4 x i32> %neg, %A
  ret <4 x i32> %and1
}

; Function Attrs: nounwind
; CHECK-LABEL: and_or
; CHECK:         xxeval v2, v2, v4, v3, 7
; CHECK-NEXT:    blr
define dso_local <4 x i32> @and_or(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C) local_unnamed_addr #0 {
entry:
  %or = or <4 x i32> %C, %B
  %and = and <4 x i32> %or, %A
  ret <4 x i32> %and
}

; Function Attrs: nounwind
; CHECK-LABEL: and_nor
; CHECK:         xxeval v2, v2, v4, v3, 8
; CHECK-NEXT:    blr
define dso_local <4 x i32> @and_nor(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C) local_unnamed_addr #0 {
entry:
  %or = or <4 x i32> %C, %B
  %neg = xor <4 x i32> %or, <i32 -1, i32 -1, i32 -1, i32 -1>
  %and = and <4 x i32> %neg, %A
  ret <4 x i32> %and
}

; Function Attrs: nounwind
; CHECK-LABEL: and_xor
; CHECK:         xxeval v2, v2, v4, v3, 6
; CHECK-NEXT:    blr
define dso_local <4 x i32> @and_xor(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C) local_unnamed_addr #0 {
entry:
  %xor = xor <4 x i32> %C, %B
  %and = and <4 x i32> %xor, %A
  ret <4 x i32> %and
}

; Function Attrs: nounwind
; CHECK-LABEL: and_eqv
; CHECK:         xxeval v2, v2, v3, v4, 9
; CHECK-NEXT:    blr
define dso_local <4 x i32> @and_eqv(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C) local_unnamed_addr #0 {
entry:
  %xor = xor <4 x i32> %B, <i32 -1, i32 -1, i32 -1, i32 -1>
  %neg = xor <4 x i32> %xor, %C
  %and = and <4 x i32> %neg, %A
  ret <4 x i32> %and
}

; Function Attrs: nounwind
; CHECK-LABEL: nand_nand
; CHECK:         xxeval v2, v2, v4, v3, 241
; CHECK-NEXT:    blr
define dso_local <4 x i32> @nand_nand(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C) local_unnamed_addr #0 {
entry:
  %and = and <4 x i32> %C, %B
  %A.not = xor <4 x i32> %A, <i32 -1, i32 -1, i32 -1, i32 -1>
  %neg2 = or <4 x i32> %and, %A.not
  ret <4 x i32> %neg2
}

; Function Attrs: nounwind
; CHECK-LABEL: nand_and
; CHECK:         xxeval v2, v3, v2, v4, 254
; CHECK-NEXT:    blr
define dso_local <4 x i32> @nand_and(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C) local_unnamed_addr #0 {
entry:
  %and = and <4 x i32> %B, %A
  %and1 = and <4 x i32> %and, %C
  %neg = xor <4 x i32> %and1, <i32 -1, i32 -1, i32 -1, i32 -1>
  ret <4 x i32> %neg
}

; Function Attrs: nounwind
; CHECK-LABEL: nand_xor
; CHECK:         xxeval v2, v2, v4, v3, 249
; CHECK-NEXT:    blr
define dso_local <4 x i32> @nand_xor(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C) local_unnamed_addr #0 {
entry:
  %xor = xor <4 x i32> %C, %B
  %and = and <4 x i32> %xor, %A
  %neg = xor <4 x i32> %and, <i32 -1, i32 -1, i32 -1, i32 -1>
  ret <4 x i32> %neg
}

; Function Attrs: nounwind
; CHECK-LABEL: nand_eqv
; CHECK:         xxeval v2, v2, v4, v3, 246
; CHECK-NEXT:    blr
define dso_local <4 x i32> @nand_eqv(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C) local_unnamed_addr #0 {
entry:
  %xor = xor <4 x i32> %C, %B
  %A.not = xor <4 x i32> %A, <i32 -1, i32 -1, i32 -1, i32 -1>
  %neg1 = or <4 x i32> %xor, %A.not
  ret <4 x i32> %neg1
}

; Function Attrs: nounwind
; CHECK-LABEL: nand_or
; CHECK:         xxeval v2, v2, v4, v3, 248
; CHECK-NEXT:    blr
define dso_local <4 x i32> @nand_or(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C) local_unnamed_addr #0 {
entry:
  %or = or <4 x i32> %C, %B
  %and = and <4 x i32> %or, %A
  %neg = xor <4 x i32> %and, <i32 -1, i32 -1, i32 -1, i32 -1>
  ret <4 x i32> %neg
}

; Function Attrs: nounwind
; CHECK-LABEL: nand_nor
; CHECK:         xxeval v2, v2, v3, v4, 247
; CHECK-NEXT:    blr
define dso_local <4 x i32> @nand_nor(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C) local_unnamed_addr #0 {
entry:
  %A.not = xor <4 x i32> %A, <i32 -1, i32 -1, i32 -1, i32 -1>
  %or = or <4 x i32> %A.not, %B
  %neg1 = or <4 x i32> %or, %C
  ret <4 x i32> %neg1
}

attributes #0 = { nounwind }
