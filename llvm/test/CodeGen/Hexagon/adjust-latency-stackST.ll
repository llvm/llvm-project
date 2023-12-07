; RUN: llc -march=hexagon < %s | FileCheck %s

; Make sure that if there's only one store to the stack, it gets packetized
; with allocframe as there's a latency of 2 cycles between allocframe and
; the following store if not in the same packet.

; CHECK: {
; CHECK: memd(r29
; CHECK-NOT: {
; CHECK: allocframe
; CHECK: }
; CHECK: = memw(gp+#G)

%struct.0 = type { ptr, i32, %struct.2 }
%struct.1 = type { i32, i32, [31 x i8] }
%struct.2 = type { %struct.1 }

@G = common global ptr null, align 4

define i32 @test(ptr nocapture %a0) #0 {
b1:
  %v2 = alloca ptr, align 4
  %v5 = load ptr, ptr %a0, align 4
  store ptr %v5, ptr %v2, align 4
  %v7 = load ptr, ptr @G, align 4
  tail call void @llvm.memcpy.p0.p0.i32(ptr align 4 %v5, ptr align 4 %v7, i32 48, i1 false)
  %v8 = getelementptr inbounds %struct.0, ptr %a0, i32 0, i32 2, i32 0, i32 1
  store i32 5, ptr %v8, align 4
  %v9 = getelementptr inbounds %struct.0, ptr %v5, i32 0, i32 2, i32 0, i32 1
  store i32 5, ptr %v9, align 4
  %v11 = load i32, ptr %a0, align 4
  store i32 %v11, ptr %v5, align 4
  %v13 = call i32 @f0(ptr nonnull %v2)
  %v14 = load ptr, ptr %v2, align 4
  %v15 = getelementptr inbounds %struct.0, ptr %v14, i32 0, i32 1
  %v16 = load i32, ptr %v15, align 4
  %v17 = icmp eq i32 %v16, 0
  br i1 %v17, label %b18, label %b32

b18:                                              ; preds = %b1
  %v20 = getelementptr inbounds %struct.0, ptr %v14, i32 0, i32 2, i32 0, i32 1
  store i32 6, ptr %v20, align 4
  %v21 = getelementptr inbounds %struct.0, ptr %a0, i32 0, i32 2, i32 0, i32 0
  %v22 = load i32, ptr %v21, align 4
  %v23 = getelementptr inbounds %struct.0, ptr %v14, i32 0, i32 2, i32 0, i32 0
  %v24 = call i32 @f1(i32 %v22, ptr %v23)
  %v25 = load ptr, ptr @G, align 4
  %v26 = load i32, ptr %v25, align 4
  %v27 = load ptr, ptr %v2, align 4
  store i32 %v26, ptr %v27, align 4
  %v28 = load ptr, ptr %v2, align 4
  %v29 = getelementptr inbounds %struct.0, ptr %v28, i32 0, i32 2, i32 0, i32 1
  %v30 = load i32, ptr %v29, align 4
  %v31 = call i32 @f2(i32 %v30, i32 10, ptr %v29)
  br label %b36

b32:                                              ; preds = %b1
  %v34 = load ptr, ptr %a0, align 4
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %a0, ptr align 4 %v34, i32 48, i1 false)
  br label %b36

b36:                                              ; preds = %b32, %b18
  ret i32 undef
}

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture writeonly, ptr nocapture readonly, i32, i1) #1

declare i32 @f0(...) #0
declare i32 @f1(...) #0
declare i32 @f2(...) #0

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }
