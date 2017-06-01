; RUN: llc -march=hexagon -mcpu=hexagonv5 -disable-hsdr -hexagon-bit=0 < %s | FileCheck %s
; CHECK: combine(r{{[0-9]+}}, r{{[0-9]+}})

@j = external global i32
@k = external global i64

define void @foo() nounwind {
entry:
  %0 = load i32, i32* @j, align 4
  %1 = load i64, i64* @k, align 8
  %conv = trunc i64 %1 to i32
  %2 = call i64 @llvm.hexagon.A2.combinew(i32 %0, i32 %conv)
  store i64 %2, i64* @k, align 8
  ret void
}

declare i64 @llvm.hexagon.A2.combinew(i32, i32) nounwind readnone
