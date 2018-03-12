; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: r{{[0-9]+}}:{{[0-9]+}} ^= pmpyw(r{{[0-9]+}},r{{[0-9]+}})

; Function Attrs: nounwind
define i32 @f0(i32 %a0, i32 %a1, i32 %a2, i32 %a3) #0 {
b0:
  %v0 = alloca i32, align 4
  %v1 = alloca i32, align 4
  %v2 = alloca i32, align 4
  %v3 = alloca i32, align 4
  %v4 = alloca i64, align 8
  %v5 = alloca i64, align 8
  store i32 %a0, i32* %v0, align 4
  store i32 %a1, i32* %v1, align 4
  store i32 %a2, i32* %v2, align 4
  store i32 %a3, i32* %v3, align 4
  %v6 = load i32, i32* %v0, align 4
  %v7 = load i32, i32* %v1, align 4
  %v8 = call i64 @llvm.hexagon.M4.pmpyw(i32 %v6, i32 %v7)
  store i64 %v8, i64* %v5, align 8
  %v9 = load i64, i64* %v5, align 8
  store i64 %v9, i64* %v4, align 8
  %v10 = load i64, i64* %v5, align 8
  %v11 = load i32, i32* %v3, align 4
  %v12 = load i64, i64* %v5, align 8
  %v13 = lshr i64 %v12, 32
  %v14 = trunc i64 %v13 to i32
  %v15 = call i64 @llvm.hexagon.M4.pmpyw.acc(i64 %v10, i32 %v11, i32 %v14)
  store i64 %v15, i64* %v5, align 8
  %v16 = load i64, i64* %v4, align 8
  %v17 = load i64, i64* %v5, align 8
  %v18 = lshr i64 %v17, 32
  %v19 = trunc i64 %v18 to i32
  %v20 = load i32, i32* %v2, align 4
  %v21 = call i64 @llvm.hexagon.M4.pmpyw.acc(i64 %v16, i32 %v19, i32 %v20)
  store i64 %v21, i64* %v4, align 8
  %v22 = load i64, i64* %v4, align 8
  %v23 = trunc i64 %v22 to i32
  ret i32 %v23
}

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.M4.pmpyw(i32, i32) #1

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.M4.pmpyw.acc(i64, i32, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
