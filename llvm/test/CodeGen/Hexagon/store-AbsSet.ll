; RUN: llc -mtriple=hexagon < %s
; REQUIRES: asserts

; Validates correct operand order for absolute-set stores.

%s.0 = type { %s.1, %s.2, %s.3, %s.3, %s.4, [8 x i8] }
%s.1 = type { i8 }
%s.2 = type { i8 }
%s.3 = type { i32 }
%s.4 = type { i32, i32, i32 }

; Function Attrs: nounwind ssp
define void @f0(ptr nocapture readonly %a0, i32 %a1) #0 {
b0:
  %v1 = load i8, ptr %a0, align 1
  %v2 = and i32 %a1, 1
  %v3 = icmp eq i32 %v2, 0
  br i1 %v3, label %b4, label %b1

b1:                                               ; preds = %b0
  %v4 = getelementptr inbounds %s.0, ptr %a0, i32 0, i32 1, i32 0
  %v5 = load i8, ptr %v4, align 1
  %v6 = icmp eq i8 %v5, 0
  br i1 %v6, label %b3, label %b2

b2:                                               ; preds = %b1
  %v7 = getelementptr %s.0, ptr %a0, i32 0, i32 2, i32 0
  %v8 = load i32, ptr %v7, align 4
  store volatile i32 %v8, ptr inttoptr (i32 -318766672 to ptr), align 16
  %v9 = getelementptr %s.0, ptr %a0, i32 0, i32 3, i32 0
  %v10 = load i32, ptr %v9, align 4
  store volatile i32 %v10, ptr inttoptr (i32 -318766672 to ptr), align 16
  br label %b3

b3:                                               ; preds = %b2, %b1
  %v11 = getelementptr inbounds %s.0, ptr %a0, i32 0, i32 4, i32 0
  %v12 = load i32, ptr %v11, align 4
  %v13 = zext i8 %v1 to i32
  %v14 = mul nsw i32 %v13, 64
  %v15 = add nsw i32 %v14, -318111684
  %v16 = inttoptr i32 %v15 to ptr
  store volatile i32 %v12, ptr %v16, align 4
  %v17 = shl i32 1, %v13
  %v18 = load volatile i32, ptr inttoptr (i32 -318111596 to ptr), align 4
  %v19 = and i32 %v17, 3
  %v20 = xor i32 %v19, 3
  %v21 = and i32 %v18, %v20
  %v22 = getelementptr inbounds %s.0, ptr %a0, i32 0, i32 4, i32 1
  %v23 = load i32, ptr %v22, align 4
  %v24 = and i32 %v23, 1
  %v25 = shl i32 %v24, %v13
  %v26 = or i32 %v25, %v21
  store volatile i32 %v26, ptr inttoptr (i32 -318111596 to ptr), align 4
  %v27 = getelementptr inbounds %s.0, ptr %a0, i32 0, i32 4, i32 2
  %v28 = load i32, ptr %v27, align 4
  %v29 = mul nsw i32 %v13, 4
  %v30 = add nsw i32 %v29, -318111592
  %v31 = inttoptr i32 %v30 to ptr
  store volatile i32 %v28, ptr %v31, align 4
  br label %b4

b4:                                               ; preds = %b3, %b0
  ret void
}

attributes #0 = { nounwind ssp }
