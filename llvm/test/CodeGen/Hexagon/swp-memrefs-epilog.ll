; RUN: llc -march=hexagon -O2 -fp-contract=fast < %s -pipeliner-experimental-cg=true | FileCheck %s

; Test that the memoperands for instructions in the epilog are updated
; correctly. Previously, the pipeliner updated the offset for the memoperands
; in the epilog. But, the value of the offset is incorrect when control flow
; branches around the kernel.

; In this test, we check that a load and store to the same location are not
; swapped due to a bad offset in the memoperands. The store and load are both
; to r29+32. If the memoperands are updated incorrectly, these are swapped.

; CHECK: [[REG0:r([0-9]+)]] = add(r29,#24)
; CHECK: memw([[REG0]]++#4) = r{{[0-9]+}}
; CHECK: r{{[0-9]+}} = memw(r29+#{{[0-9]+}})

%s.0 = type { %s.1 }
%s.1 = type { %s.2 }
%s.2 = type { %s.3 }
%s.3 = type { [9 x float] }
%s.4 = type { %s.5 }
%s.5 = type { %s.6 }
%s.6 = type { %s.7 }
%s.7 = type { [3 x float] }

@g0 = external hidden unnamed_addr constant [29 x i8], align 1

define i32 @f0() unnamed_addr {
b0:
  %v0 = alloca %s.0, align 4
  %v1 = alloca %s.4, align 4
  %v2 = bitcast ptr %v0 to ptr
  call void @llvm.lifetime.start.p0(i64 36, ptr %v2)
  %v3 = getelementptr inbounds %s.0, ptr %v0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store float 0x3FEFFF7160000000, ptr %v3, align 4
  %v4 = getelementptr inbounds %s.0, ptr %v0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1
  store float 0xBF87867F00000000, ptr %v4, align 4
  %v5 = getelementptr inbounds %s.0, ptr %v0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 2
  store float 0xBF6185CEE0000000, ptr %v5, align 4
  %v6 = getelementptr inbounds %s.0, ptr %v0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 3
  store float 0x3F8780BAA0000000, ptr %v6, align 4
  %v7 = getelementptr inbounds %s.0, ptr %v0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4
  store float 0x3FEFFF5C60000000, ptr %v7, align 4
  %v8 = getelementptr inbounds %s.0, ptr %v0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 5
  store float 0xBF74717160000000, ptr %v8, align 4
  %v9 = getelementptr inbounds %s.0, ptr %v0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 6
  store float 0x3F61FF7160000000, ptr %v9, align 4
  %v10 = getelementptr inbounds %s.0, ptr %v0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 7
  store float 0x3F74573A80000000, ptr %v10, align 4
  %v11 = getelementptr inbounds %s.0, ptr %v0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8
  store float 0x3FEFFFE080000000, ptr %v11, align 4
  %v12 = bitcast ptr %v1 to ptr
  call void @llvm.lifetime.start.p0(i64 12, ptr %v12)
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v13 = phi i32 [ 0, %b0 ], [ %v29, %b1 ]
  %v14 = mul nuw nsw i32 %v13, 3
  %v15 = getelementptr inbounds %s.0, ptr %v0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 %v14
  %v16 = load float, ptr %v15, align 4
  %v17 = fmul float %v16, 0x3FE7B2B120000000
  %v18 = add nuw nsw i32 %v14, 1
  %v19 = getelementptr inbounds %s.0, ptr %v0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 %v18
  %v20 = load float, ptr %v19, align 4
  %v21 = fmul float %v20, 0x3FDA8BC9C0000000
  %v22 = fsub float %v21, %v17
  %v23 = add nuw nsw i32 %v14, 2
  %v24 = getelementptr inbounds %s.0, ptr %v0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 %v23
  %v25 = load float, ptr %v24, align 4
  %v26 = fmul float %v25, 0x40030D6700000000
  %v27 = fadd float %v22, %v26
  %v28 = getelementptr inbounds %s.4, ptr %v1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 %v13
  store float %v27, ptr %v28, align 4
  %v29 = add nuw nsw i32 %v13, 1
  %v30 = icmp eq i32 %v29, 3
  br i1 %v30, label %b2, label %b1

b2:                                               ; preds = %b1
  %v31 = getelementptr inbounds %s.4, ptr %v1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %v32 = load float, ptr %v31, align 4
  %v33 = fpext float %v32 to double
  %v34 = getelementptr inbounds %s.4, ptr %v1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1
  %v35 = load float, ptr %v34, align 4
  %v36 = fpext float %v35 to double
  %v37 = getelementptr inbounds %s.4, ptr %v1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 2
  %v38 = load float, ptr %v37, align 4
  %v39 = fpext float %v38 to double
  %v40 = tail call i32 (ptr, ...) @f1(ptr @g0, double %v33, double %v36, double %v39)
  call void @llvm.lifetime.end.p0(i64 12, ptr nonnull %v12)
  call void @llvm.lifetime.end.p0(i64 36, ptr nonnull %v2)
  ret i32 0
}

declare i32 @f1(ptr nocapture readonly, ...) local_unnamed_addr

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #0

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #0

attributes #0 = { argmemonly nounwind }
