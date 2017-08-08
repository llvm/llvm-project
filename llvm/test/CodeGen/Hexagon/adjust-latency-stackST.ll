; RUN: llc -march=hexagon -disable-post-ra < %s | FileCheck %s

; Make sure that if there's only one store to the stack, it gets packetized
; with allocframe as there's a latency of 2 cycles between allocframe and
; the following store if not in the same packet.

; CHECK: {
; CHECK: memd(r29
; CHECK-NOT: {
; CHECK: allocframe
; CHECK: }
; CHECK: = memw(gp+#G)

%struct.0 = type { %struct.0*, i32, %struct.2 }
%struct.1 = type { i32, i32, [31 x i8] }
%struct.2 = type { %struct.1 }

@G = common global %struct.0* null, align 4

define i32 @test(%struct.0* nocapture %a0) #0 {
b1:
  %v2 = alloca %struct.0*, align 4
  %v3 = bitcast %struct.0** %v2 to i8*
  %v4 = getelementptr inbounds %struct.0, %struct.0* %a0, i32 0, i32 0
  %v5 = load %struct.0*, %struct.0** %v4, align 4
  store %struct.0* %v5, %struct.0** %v2, align 4
  %v6 = bitcast %struct.0* %v5 to i8*
  %v7 = load i8*, i8** bitcast (%struct.0** @G to i8**), align 4
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %v6, i8* %v7, i32 48, i32 4, i1 false)
  %v8 = getelementptr inbounds %struct.0, %struct.0* %a0, i32 0, i32 2, i32 0, i32 1
  store i32 5, i32* %v8, align 4
  %v9 = getelementptr inbounds %struct.0, %struct.0* %v5, i32 0, i32 2, i32 0, i32 1
  store i32 5, i32* %v9, align 4
  %v10 = bitcast %struct.0* %a0 to i32*
  %v11 = load i32, i32* %v10, align 4
  %v12 = bitcast %struct.0* %v5 to i32*
  store i32 %v11, i32* %v12, align 4
  %v13 = call i32 bitcast (i32 (...)* @f0 to i32 (%struct.0**)*)(%struct.0** nonnull %v2)
  %v14 = load %struct.0*, %struct.0** %v2, align 4
  %v15 = getelementptr inbounds %struct.0, %struct.0* %v14, i32 0, i32 1
  %v16 = load i32, i32* %v15, align 4
  %v17 = icmp eq i32 %v16, 0
  br i1 %v17, label %b18, label %b32

b18:                                              ; preds = %b1
  %v19 = bitcast %struct.0** %v2 to i32**
  %v20 = getelementptr inbounds %struct.0, %struct.0* %v14, i32 0, i32 2, i32 0, i32 1
  store i32 6, i32* %v20, align 4
  %v21 = getelementptr inbounds %struct.0, %struct.0* %a0, i32 0, i32 2, i32 0, i32 0
  %v22 = load i32, i32* %v21, align 4
  %v23 = getelementptr inbounds %struct.0, %struct.0* %v14, i32 0, i32 2, i32 0, i32 0
  %v24 = call i32 bitcast (i32 (...)* @f1 to i32 (i32, i32*)*)(i32 %v22, i32* %v23)
  %v25 = load i32*, i32** bitcast (%struct.0** @G to i32**), align 4
  %v26 = load i32, i32* %v25, align 4
  %v27 = load i32*, i32** %v19, align 4
  store i32 %v26, i32* %v27, align 4
  %v28 = load %struct.0*, %struct.0** %v2, align 4
  %v29 = getelementptr inbounds %struct.0, %struct.0* %v28, i32 0, i32 2, i32 0, i32 1
  %v30 = load i32, i32* %v29, align 4
  %v31 = call i32 bitcast (i32 (...)* @f2 to i32 (i32, i32, i32*)*)(i32 %v30, i32 10, i32* %v29)
  br label %b36

b32:                                              ; preds = %b1
  %v33 = bitcast %struct.0* %a0 to i8**
  %v34 = load i8*, i8** %v33, align 4
  %v35 = bitcast %struct.0* %a0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %v35, i8* %v34, i32 48, i32 4, i1 false)
  br label %b36

b36:                                              ; preds = %b32, %b18
  ret i32 undef
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i32, i1) #1

declare i32 @f0(...) #0
declare i32 @f1(...) #0
declare i32 @f2(...) #0

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }
