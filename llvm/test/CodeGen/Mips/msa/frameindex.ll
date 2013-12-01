; RUN: llc -march=mips -mattr=+msa,+fp64 < %s | FileCheck -check-prefix=MIPS32-AE -check-prefix=MIPS32-BE %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64 < %s | FileCheck -check-prefix=MIPS32-AE -check-prefix=MIPS32-LE %s

define void @loadstore_v16i8_near() nounwind {
  ; MIPS32-AE: loadstore_v16i8_near:

  %1 = alloca <16 x i8>
  %2 = load volatile <16 x i8>* %1
  ; MIPS32-AE: ld.b [[R1:\$w[0-9]+]], 0($sp)
  store volatile <16 x i8> %2, <16 x i8>* %1
  ; MIPS32-AE: st.b [[R1]], 0($sp)

  ret void
  ; MIPS32-AE: .size loadstore_v16i8_near
}

define void @loadstore_v16i8_just_under_simm10() nounwind {
  ; MIPS32-AE: loadstore_v16i8_just_under_simm10:

  %1 = alloca <16 x i8>
  %2 = alloca [496 x i8] ; Push the frame right up to 512 bytes

  %3 = load volatile <16 x i8>* %1
  ; MIPS32-AE: ld.b [[R1:\$w[0-9]+]], 496($sp)
  store volatile <16 x i8> %3, <16 x i8>* %1
  ; MIPS32-AE: st.b [[R1]], 496($sp)

  ret void
  ; MIPS32-AE: .size loadstore_v16i8_just_under_simm10
}

define void @loadstore_v16i8_just_over_simm10() nounwind {
  ; MIPS32-AE: loadstore_v16i8_just_over_simm10:

  %1 = alloca <16 x i8>
  %2 = alloca [497 x i8] ; Push the frame just over 512 bytes

  %3 = load volatile <16 x i8>* %1
  ; MIPS32-AE: addiu [[BASE:\$[0-9]+]], $sp, 512
  ; MIPS32-AE: ld.b [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <16 x i8> %3, <16 x i8>* %1
  ; MIPS32-AE: addiu [[BASE:\$[0-9]+]], $sp, 512
  ; MIPS32-AE: st.b [[R1]], 0([[BASE]])

  ret void
  ; MIPS32-AE: .size loadstore_v16i8_just_over_simm10
}

define void @loadstore_v16i8_just_under_simm16() nounwind {
  ; MIPS32-AE: loadstore_v16i8_just_under_simm16:

  %1 = alloca <16 x i8>
  %2 = alloca [32752 x i8] ; Push the frame right up to 32768 bytes

  %3 = load volatile <16 x i8>* %1
  ; MIPS32-AE: ori [[R2:\$[0-9]+]], $zero, 32768
  ; MIPS32-AE: addu [[BASE:\$[0-9]+]], $sp, [[R2]]
  ; MIPS32-AE: ld.b [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <16 x i8> %3, <16 x i8>* %1
  ; MIPS32-AE: ori [[R2:\$[0-9]+]], $zero, 32768
  ; MIPS32-AE: addu [[BASE:\$[0-9]+]], $sp, [[R2]]
  ; MIPS32-AE: st.b [[R1]], 0([[BASE]])

  ret void
  ; MIPS32-AE: .size loadstore_v16i8_just_under_simm16
}

define void @loadstore_v16i8_just_over_simm16() nounwind {
  ; MIPS32-AE: loadstore_v16i8_just_over_simm16:

  %1 = alloca <16 x i8>
  %2 = alloca [32753 x i8] ; Push the frame just over 32768 bytes

  %3 = load volatile <16 x i8>* %1
  ; MIPS32-AE: ori [[R2:\$[0-9]+]], $zero, 32768
  ; MIPS32-AE: addu [[BASE:\$[0-9]+]], $sp, [[R2]]
  ; MIPS32-AE: ld.b [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <16 x i8> %3, <16 x i8>* %1
  ; MIPS32-AE: ori [[R2:\$[0-9]+]], $zero, 32768
  ; MIPS32-AE: addu [[BASE:\$[0-9]+]], $sp, [[R2]]
  ; MIPS32-AE: st.b [[R1]], 0([[BASE]])

  ret void
  ; MIPS32-AE: .size loadstore_v16i8_just_over_simm16
}
