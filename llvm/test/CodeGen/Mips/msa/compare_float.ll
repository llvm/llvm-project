; RUN: llc -mtriple=mips-elf -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s
; RUN: llc -mtriple=mipsel-elf -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s

declare <4 x float> @llvm.mips.fmax.w(<4 x float>, <4 x float>) nounwind
declare <2 x double> @llvm.mips.fmax.d(<2 x double>, <2 x double>) nounwind
declare <4 x float> @llvm.mips.fmin.w(<4 x float>, <4 x float>) nounwind
declare <2 x double> @llvm.mips.fmin.d(<2 x double>, <2 x double>) nounwind

define void @false_v4f32(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: false_v4f32:

  %1 = load <4 x float>, ptr %a
  %2 = load <4 x float>, ptr %b
  %3 = fcmp false <4 x float> %1, %2
  %4 = sext <4 x i1> %3 to <4 x i32>
  store <4 x i32> %4, ptr %c
  ret void

  ; (setcc $a, $b, SETFALSE) is always folded, so we won't get fcaf:
  ; CHECK-DAG: ldi.b [[R1:\$w[0-9]+]], 0
  ; CHECK-DAG: st.w [[R1]], 0($4)
  ; CHECK: .size false_v4f32
}

define void @false_v2f64(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: false_v2f64:

  %1 = load <2 x double>, ptr %a
  %2 = load <2 x double>, ptr %b
  %3 = fcmp false <2 x double> %1, %2
  %4 = sext <2 x i1> %3 to <2 x i64>
  store <2 x i64> %4, ptr %c
  ret void

  ; (setcc $a, $b, SETFALSE) is always folded
  ; CHECK-DAG: ldi.b [[R1:\$w[0-9]+]], 0
  ; CHECK-DAG: st.w [[R1]], 0($4)
  ; CHECK: .size false_v2f64
}

define void @oeq_v4f32(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: oeq_v4f32:

  %1 = load <4 x float>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x float>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp oeq <4 x float> %1, %2
  %4 = sext <4 x i1> %3 to <4 x i32>
  ; CHECK-DAG: fceq.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size oeq_v4f32
}

define void @oeq_v2f64(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: oeq_v2f64:

  %1 = load <2 x double>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x double>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp oeq <2 x double> %1, %2
  %4 = sext <2 x i1> %3 to <2 x i64>
  ; CHECK-DAG: fceq.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size oeq_v2f64
}

define void @oge_v4f32(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: oge_v4f32:

  %1 = load <4 x float>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x float>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp oge <4 x float> %1, %2
  %4 = sext <4 x i1> %3 to <4 x i32>
  ; CHECK-DAG: fcle.w [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <4 x i32> %4, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size oge_v4f32
}

define void @oge_v2f64(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: oge_v2f64:

  %1 = load <2 x double>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x double>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp oge <2 x double> %1, %2
  %4 = sext <2 x i1> %3 to <2 x i64>
  ; CHECK-DAG: fcle.d [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <2 x i64> %4, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size oge_v2f64
}

define void @ogt_v4f32(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: ogt_v4f32:

  %1 = load <4 x float>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x float>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp ogt <4 x float> %1, %2
  %4 = sext <4 x i1> %3 to <4 x i32>
  ; CHECK-DAG: fclt.w [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <4 x i32> %4, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size ogt_v4f32
}

define void @ogt_v2f64(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: ogt_v2f64:

  %1 = load <2 x double>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x double>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp ogt <2 x double> %1, %2
  %4 = sext <2 x i1> %3 to <2 x i64>
  ; CHECK-DAG: fclt.d [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <2 x i64> %4, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size ogt_v2f64
}

define void @ole_v4f32(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: ole_v4f32:

  %1 = load <4 x float>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x float>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp ole <4 x float> %1, %2
  %4 = sext <4 x i1> %3 to <4 x i32>
  ; CHECK-DAG: fcle.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size ole_v4f32
}

define void @ole_v2f64(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: ole_v2f64:

  %1 = load <2 x double>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x double>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp ole <2 x double> %1, %2
  %4 = sext <2 x i1> %3 to <2 x i64>
  ; CHECK-DAG: fcle.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size ole_v2f64
}

define void @olt_v4f32(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: olt_v4f32:

  %1 = load <4 x float>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x float>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp olt <4 x float> %1, %2
  %4 = sext <4 x i1> %3 to <4 x i32>
  ; CHECK-DAG: fclt.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size olt_v4f32
}

define void @olt_v2f64(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: olt_v2f64:

  %1 = load <2 x double>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x double>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp olt <2 x double> %1, %2
  %4 = sext <2 x i1> %3 to <2 x i64>
  ; CHECK-DAG: fclt.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size olt_v2f64
}

define void @one_v4f32(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: one_v4f32:

  %1 = load <4 x float>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x float>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp one <4 x float> %1, %2
  %4 = sext <4 x i1> %3 to <4 x i32>
  ; CHECK-DAG: fcne.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size one_v4f32
}

define void @one_v2f64(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: one_v2f64:

  %1 = load <2 x double>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x double>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp one <2 x double> %1, %2
  %4 = sext <2 x i1> %3 to <2 x i64>
  ; CHECK-DAG: fcne.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size one_v2f64
}

define void @ord_v4f32(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: ord_v4f32:

  %1 = load <4 x float>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x float>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp ord <4 x float> %1, %2
  %4 = sext <4 x i1> %3 to <4 x i32>
  ; CHECK-DAG: fcor.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size ord_v4f32
}

define void @ord_v2f64(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: ord_v2f64:

  %1 = load <2 x double>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x double>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp ord <2 x double> %1, %2
  %4 = sext <2 x i1> %3 to <2 x i64>
  ; CHECK-DAG: fcor.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size ord_v2f64
}

define void @ueq_v4f32(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: ueq_v4f32:

  %1 = load <4 x float>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x float>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp ueq <4 x float> %1, %2
  %4 = sext <4 x i1> %3 to <4 x i32>
  ; CHECK-DAG: fcueq.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size ueq_v4f32
}

define void @ueq_v2f64(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: ueq_v2f64:

  %1 = load <2 x double>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x double>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp ueq <2 x double> %1, %2
  %4 = sext <2 x i1> %3 to <2 x i64>
  ; CHECK-DAG: fcueq.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size ueq_v2f64
}

define void @uge_v4f32(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: uge_v4f32:

  %1 = load <4 x float>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x float>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp uge <4 x float> %1, %2
  %4 = sext <4 x i1> %3 to <4 x i32>
  ; CHECK-DAG: fcule.w [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <4 x i32> %4, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size uge_v4f32
}

define void @uge_v2f64(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: uge_v2f64:

  %1 = load <2 x double>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x double>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp uge <2 x double> %1, %2
  %4 = sext <2 x i1> %3 to <2 x i64>
  ; CHECK-DAG: fcule.d [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <2 x i64> %4, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size uge_v2f64
}

define void @ugt_v4f32(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: ugt_v4f32:

  %1 = load <4 x float>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x float>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp ugt <4 x float> %1, %2
  %4 = sext <4 x i1> %3 to <4 x i32>
  ; CHECK-DAG: fcult.w [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <4 x i32> %4, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size ugt_v4f32
}

define void @ugt_v2f64(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: ugt_v2f64:

  %1 = load <2 x double>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x double>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp ugt <2 x double> %1, %2
  %4 = sext <2 x i1> %3 to <2 x i64>
  ; CHECK-DAG: fcult.d [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <2 x i64> %4, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size ugt_v2f64
}

define void @ule_v4f32(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: ule_v4f32:

  %1 = load <4 x float>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x float>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp ule <4 x float> %1, %2
  %4 = sext <4 x i1> %3 to <4 x i32>
  ; CHECK-DAG: fcule.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size ule_v4f32
}

define void @ule_v2f64(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: ule_v2f64:

  %1 = load <2 x double>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x double>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp ule <2 x double> %1, %2
  %4 = sext <2 x i1> %3 to <2 x i64>
  ; CHECK-DAG: fcule.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size ule_v2f64
}

define void @ult_v4f32(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: ult_v4f32:

  %1 = load <4 x float>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x float>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp ult <4 x float> %1, %2
  %4 = sext <4 x i1> %3 to <4 x i32>
  ; CHECK-DAG: fcult.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size ult_v4f32
}

define void @ult_v2f64(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: ult_v2f64:

  %1 = load <2 x double>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x double>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp ult <2 x double> %1, %2
  %4 = sext <2 x i1> %3 to <2 x i64>
  ; CHECK-DAG: fcult.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size ult_v2f64
}

define void @uno_v4f32(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: uno_v4f32:

  %1 = load <4 x float>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x float>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp uno <4 x float> %1, %2
  %4 = sext <4 x i1> %3 to <4 x i32>
  ; CHECK-DAG: fcun.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size uno_v4f32
}

define void @uno_v2f64(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: uno_v2f64:

  %1 = load <2 x double>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x double>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp uno <2 x double> %1, %2
  %4 = sext <2 x i1> %3 to <2 x i64>
  ; CHECK-DAG: fcun.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size uno_v2f64
}

define void @true_v4f32(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: true_v4f32:

  %1 = load <4 x float>, ptr %a
  %2 = load <4 x float>, ptr %b
  %3 = fcmp true <4 x float> %1, %2
  %4 = sext <4 x i1> %3 to <4 x i32>
  store <4 x i32> %4, ptr %c
  ret void

  ; (setcc $a, $b, SETTRUE) is always folded, so we won't get fcaf:
  ; CHECK-DAG: ldi.b [[R1:\$w[0-9]+]], -1
  ; CHECK-DAG: st.w [[R1]], 0($4)
  ; CHECK: .size true_v4f32
}

define void @true_v2f64(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: true_v2f64:

  %1 = load <2 x double>, ptr %a
  %2 = load <2 x double>, ptr %b
  %3 = fcmp true <2 x double> %1, %2
  %4 = sext <2 x i1> %3 to <2 x i64>
  store <2 x i64> %4, ptr %c
  ret void

  ; (setcc $a, $b, SETTRUE) is always folded.
  ; CHECK-DAG: ldi.b [[R1:\$w[0-9]+]], -1
  ; CHECK-DAG: st.w [[R1]], 0($4)
  ; CHECK: .size true_v2f64
}

define void @bsel_v4f32(ptr %d, ptr %a, ptr %b,
                          ptr %c) nounwind {
  ; CHECK: bsel_v4f32:

  %1 = load <4 x float>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x float>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = load <4 x float>, ptr %c
  ; CHECK-DAG: ld.w [[R3:\$w[0-9]+]], 0($7)
  %4 = fcmp ogt <4 x float> %1, %2
  ; CHECK-DAG: fclt.w [[R4:\$w[0-9]+]], [[R2]], [[R1]]
  %5 = select <4 x i1> %4, <4 x float> %1, <4 x float> %3
  ; Note that IfSet and IfClr are swapped since the condition is inverted
  ; CHECK-DAG: bsel.v [[R4]], [[R3]], [[R1]]
  store <4 x float> %5, ptr %d
  ; CHECK-DAG: st.w [[R4]], 0($4)

  ret void
  ; CHECK: .size bsel_v4f32
}

define void @bsel_v2f64(ptr %d, ptr %a, ptr %b,
                          ptr %c) nounwind {
  ; CHECK: bsel_v2f64:

  %1 = load <2 x double>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x double>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = load <2 x double>, ptr %c
  ; CHECK-DAG: ld.d [[R3:\$w[0-9]+]], 0($7)
  %4 = fcmp ogt <2 x double> %1, %2
  ; CHECK-DAG: fclt.d [[R4:\$w[0-9]+]], [[R2]], [[R1]]
  %5 = select <2 x i1> %4, <2 x double> %1, <2 x double> %3
  ; Note that IfSet and IfClr are swapped since the condition is inverted
  ; CHECK-DAG: bsel.v [[R4]], [[R3]], [[R1]]
  store <2 x double> %5, ptr %d
  ; CHECK-DAG: st.d [[R4]], 0($4)

  ret void
  ; CHECK: .size bsel_v2f64
}

define void @bseli_v4f32(ptr %d, ptr %a, ptr %b,
                          ptr %c) nounwind {
  ; CHECK: bseli_v4f32:

  %1 = load <4 x float>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x float>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp ogt <4 x float> %1, %2
  ; CHECK-DAG: fclt.w [[R4:\$w[0-9]+]], [[R2]], [[R1]]
  %4 = select <4 x i1> %3, <4 x float> %1, <4 x float> zeroinitializer
  ; Note that IfSet and IfClr are swapped since the condition is inverted
  ; CHECK-DAG: bsel.v [[R4]], [[R3:\$w[0-9]+]], [[R1]]
  store <4 x float> %4, ptr %d
  ; CHECK-DAG: st.w [[R4]], 0($4)

  ret void
  ; CHECK: .size bseli_v4f32
}

define void @bseli_v2f64(ptr %d, ptr %a, ptr %b,
                          ptr %c) nounwind {
  ; CHECK: bseli_v2f64:

  %1 = load <2 x double>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x double>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = fcmp ogt <2 x double> %1, %2
  ; CHECK-DAG: fclt.d [[R4:\$w[0-9]+]], [[R2]], [[R1]]
  %4 = select <2 x i1> %3, <2 x double> %1, <2 x double> zeroinitializer
  ; Note that IfSet and IfClr are swapped since the condition is inverted
  ; CHECK-DAG: bsel.v [[R4]], [[R3:\$w[0-9]+]], [[R1]]
  store <2 x double> %4, ptr %d
  ; CHECK-DAG: st.d [[R4]], 0($4)

  ret void
  ; CHECK: .size bseli_v2f64
}

define void @max_v4f32(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: max_v4f32:

  %1 = load <4 x float>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x float>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = tail call <4 x float> @llvm.mips.fmax.w(<4 x float> %1, <4 x float> %2)
  ; CHECK-DAG: fmax.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x float> %3, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size max_v4f32
}

define void @max_v2f64(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: max_v2f64:

  %1 = load <2 x double>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x double>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = tail call <2 x double> @llvm.mips.fmax.d(<2 x double> %1, <2 x double> %2)
  ; CHECK-DAG: fmax.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x double> %3, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size max_v2f64
}

define void @min_v4f32(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: min_v4f32:

  %1 = load <4 x float>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x float>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = tail call <4 x float> @llvm.mips.fmin.w(<4 x float> %1, <4 x float> %2)
  ; CHECK-DAG: fmin.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x float> %3, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size min_v4f32
}

define void @min_v2f64(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK: min_v2f64:

  %1 = load <2 x double>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x double>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = tail call <2 x double> @llvm.mips.fmin.d(<2 x double> %1, <2 x double> %2)
  ; CHECK-DAG: fmin.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x double> %3, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size min_v2f64
}
