; RUN: llc -march=hexagon -hexagon-expand-condsets=0 < %s | FileCheck %s

; CHECK-DAG: r{{[0-9]+:[0-9]+}} = add(r{{[0-9]+:[0-9]+}},r{{[0-9]+:[0-9]+}})
; CHECK-DAG: r{{[0-9]+:[0-9]+}} = add(r{{[0-9]+:[0-9]+}},r{{[0-9]+:[0-9]+}})
; CHECK-DAG: p{{[0-9]+}} = cmp.gtu(r{{[0-9]+:[0-9]+}},r{{[0-9]+:[0-9]+}})
; CHECK-DAG: p{{[0-9]+}} = cmp.gtu(r{{[0-9]+:[0-9]+}},r{{[0-9]+:[0-9]+}})
; CHECK-DAG: r{{[0-9]+}} = mux(p{{[0-9]+}},r{{[0-9]+}},r{{[0-9]+}})
; CHECK-DAG: r{{[0-9]+}} = mux(p{{[0-9]+}},r{{[0-9]+}},r{{[0-9]+}})

define void @check_adde_addc(i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64* %a4, i64* %a5) {
b6:
  %v7 = zext i64 %a0 to i128
  %v8 = zext i64 %a1 to i128
  %v9 = shl i128 %v8, 64
  %v10 = or i128 %v7, %v9
  %v11 = zext i64 %a2 to i128
  %v12 = zext i64 %a3 to i128
  %v13 = shl i128 %v12, 64
  %v14 = or i128 %v11, %v13
  %v15 = add i128 %v10, %v14
  %v16 = lshr i128 %v15, 64
  %v17 = trunc i128 %v15 to i64
  %v18 = trunc i128 %v16 to i64
  store i64 %v17, i64* %a4
  store i64 %v18, i64* %a5
  ret void
}
