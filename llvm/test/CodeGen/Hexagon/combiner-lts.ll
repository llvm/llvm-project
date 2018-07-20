; RUN: llc -march=hexagon -O2 < %s | FileCheck %s

; In DAG combiner, eliminate a store in cases where the store is fed by a
; load from the same location.  This is already done in cases where the store's
; chain reaches the "output chain" of the load, this tests for cases where
; the load's "input chain" is reached via an intervening node (eg. TokenFactor)
; that ensures ordering.

target triple = "hexagon"

%s.0 = type { [3 x i32] }

; Function Attrs: nounwind
define void @f0(i32 %a0, i32 %a1, %s.0* nocapture %a2, %s.0* nocapture %a3) #0 {
b0:
; Pick one store that happens as a result.  This isn't the best, but a regular
; expression for a register name matches some unrelated load.
; CHECK: %bb.
; CHECK: = memw(r3+#8)
; CHECK-NOT: memw(r3+#8) =
; CHECK: %bb.
  %v0 = bitcast %s.0* %a2 to i8*
  %v1 = bitcast %s.0* %a3 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %v0, i8* align 4 %v1, i32 12, i1 false)
  %v2 = bitcast %s.0* %a2 to i96*
  %v3 = zext i32 %a0 to i96
  %v4 = load i96, i96* %v2, align 4
  %v5 = shl nuw nsw i96 %v3, 48
  %v6 = and i96 %v5, 281474976710656
  %v7 = and i96 %v4, -281474976710657
  %v8 = or i96 %v7, %v6
  store i96 %v8, i96* %v2, align 4
  %v9 = icmp eq i32 %a1, 2147483647
  br i1 %v9, label %b1, label %b2

b1:                                               ; preds = %b0
  %v10 = and i96 %v8, -12582913
  br label %b3

b2:                                               ; preds = %b0
  %v11 = bitcast %s.0* %a3 to i96*
  %v12 = load i96, i96* %v11, align 4
  %v13 = trunc i96 %v12 to i32
  %v14 = add i32 %v13, %a1
  %v15 = zext i32 %v14 to i96
  %v16 = and i96 %v15, 4194303
  %v17 = and i96 %v8, -4194304
  %v18 = or i96 %v16, %v17
  store i96 %v18, i96* %v2, align 4
  %v19 = load i96, i96* %v11, align 4
  %v20 = and i96 %v19, 12582912
  %v21 = and i96 %v18, -12582913
  %v22 = or i96 %v21, %v20
  br label %b3

b3:                                               ; preds = %b2, %b1
  %v23 = phi i96 [ %v22, %b2 ], [ %v10, %b1 ]
  store i96 %v23, i96* %v2, align 4
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i1) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }
