; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s

define void @ashr_v4i32(<4 x i32>* %c) nounwind {
  ; CHECK-LABEL: ashr_v4i32:

  %1 = ashr <4 x i32> <i32 1, i32 2, i32 4, i32 8>,
                      <i32 0, i32 1, i32 2, i32 3>
  ; CHECK-NOT: sra
  ; CHECK-DAG: ldi.w [[R1:\$w[0-9]+]], 1
  ; CHECK-NOT: sra
  store volatile <4 x i32> %1, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R1]], 0($4)

  %2 = ashr <4 x i32> <i32 -2, i32 -4, i32 -8, i32 -16>,
                      <i32 0, i32 1, i32 2, i32 3>
  ; CHECK-NOT: sra
  ; CHECK-DAG: ldi.w [[R1:\$w[0-9]+]], -2
  ; CHECK-NOT: sra
  store volatile <4 x i32> %2, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R1]], 0($4)

  ret void
  ; CHECK-LABEL: .size ashr_v4i32
}

define void @lshr_v4i32(<4 x i32>* %c) nounwind {
  ; CHECK-LABEL: lshr_v4i32:

  %1 = lshr <4 x i32> <i32 1, i32 2, i32 4, i32 8>,
                      <i32 0, i32 1, i32 2, i32 3>
  ; CHECK-NOT: srl
  ; CHECK-DAG: ldi.w [[R1:\$w[0-9]+]], 1
  ; CHECK-NOT: srl
  store volatile <4 x i32> %1, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R1]], 0($4)

  %2 = lshr <4 x i32> <i32 -2, i32 -4, i32 -8, i32 -16>,
                      <i32 0, i32 1, i32 2, i32 3>
  ; CHECK-NOT: srl
  ; CHECK-DAG: addiu [[CPOOL:\$[0-9]+]], {{.*}}, %lo($
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0([[CPOOL]])
  ; CHECK-NOT: srl
  store volatile <4 x i32> %2, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R1]], 0($4)

  ret void
  ; CHECK-LABEL: .size lshr_v4i32
}

define void @shl_v4i32(<4 x i32>* %c) nounwind {
  ; CHECK-LABEL: shl_v4i32:

  %1 = shl <4 x i32> <i32 8, i32 4, i32 2, i32 1>,
                     <i32 0, i32 1, i32 2, i32 3>
  ; CHECK-NOT: sll
  ; CHECK-DAG: ldi.w [[R1:\$w[0-9]+]], 8
  ; CHECK-NOT: sll
  store volatile <4 x i32> %1, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R1]], 0($4)

  %2 = shl <4 x i32> <i32 -8, i32 -4, i32 -2, i32 -1>,
                     <i32 0, i32 1, i32 2, i32 3>
  ; CHECK-NOT: sll
  ; CHECK-DAG: ldi.w [[R1:\$w[0-9]+]], -8
  ; CHECK-NOT: sll
  store volatile <4 x i32> %2, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R1]], 0($4)

  ret void
  ; CHECK-LABEL: .size shl_v4i32
}
