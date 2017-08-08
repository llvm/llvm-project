; RUN: llc -march=hexagon < %s | FileCheck %s

target triple = "hexagon"

; CHECK-LABEL: test1:
; CHECK: [[REG1:(r[0-9]+)]] = ##875770417
; CHECK-DAG: memw(r29+#4) = [[REG1]]
; CHECK-DAG: memw(r29+#8) = #51
; CHECK-DAG: memh(r29+#12) = #50
; CHECK-DAG: memb(r29+#15) = #49
define void @test1() {
b0:
  %v1 = alloca [1 x i8], align 1
  %v2 = alloca i16, align 2
  %v3 = alloca i32, align 4
  %v4 = alloca i32, align 4
  %v5 = getelementptr inbounds [1 x i8], [1 x i8]* %v1, i32 0, i32 0
  call void @llvm.lifetime.start(i64 1, i8* %v5)
  store i8 49, i8* %v5, align 1
  %v6 = bitcast i16* %v2 to i8*
  call void @llvm.lifetime.start(i64 2, i8* %v6)
  store i16 50, i16* %v2, align 2
  %v7 = bitcast i32* %v3 to i8*
  call void @llvm.lifetime.start(i64 4, i8* %v7)
  store i32 51, i32* %v3, align 4
  %v8 = bitcast i32* %v4 to i8*
  call void @llvm.lifetime.start(i64 4, i8* %v8)
  store i32 875770417, i32* %v4, align 4
  call void @test4(i8* %v5, i8* %v6, i8* %v7, i8* %v8)
  call void @llvm.lifetime.end(i64 4, i8* %v8)
  call void @llvm.lifetime.end(i64 4, i8* %v7)
  call void @llvm.lifetime.end(i64 2, i8* %v6)
  call void @llvm.lifetime.end(i64 1, i8* %v5)
  ret void
}

; CHECK-LABEL: test2:
; CHECK-DAG: memw(r29+#208) = #51
; CHECK-DAG: memh(r29+#212) = r{{[0-9]+}}
; CHECK-DAG: memb(r29+#215) = r{{[0-9]+}}
define void @test2() {
b0:
  %v1 = alloca [1 x i8], align 1
  %v2 = alloca i16, align 2
  %v3 = alloca i32, align 4
  %v4 = alloca i32, align 4
  %v5 = alloca [100 x i8], align 8
  %v6 = alloca [101 x i8], align 8
  %v7 = getelementptr inbounds [1 x i8], [1 x i8]* %v1, i32 0, i32 0
  call void @llvm.lifetime.start(i64 1, i8* %v7)
  store i8 49, i8* %v7, align 1
  %v8 = bitcast i16* %v2 to i8*
  call void @llvm.lifetime.start(i64 2, i8* %v8)
  store i16 50, i16* %v2, align 2
  %v9 = bitcast i32* %v3 to i8*
  call void @llvm.lifetime.start(i64 4, i8* %v9)
  store i32 51, i32* %v3, align 4
  %v10 = bitcast i32* %v4 to i8*
  call void @llvm.lifetime.start(i64 4, i8* %v10)
  store i32 875770417, i32* %v4, align 4
  %v11 = getelementptr inbounds [100 x i8], [100 x i8]* %v5, i32 0, i32 0
  call void @llvm.lifetime.start(i64 100, i8* %v11)
  call void @llvm.memset.p0i8.i32(i8* %v11, i8 0, i32 100, i32 8, i1 false)
  store i8 50, i8* %v11, align 8
  %v12 = getelementptr inbounds [101 x i8], [101 x i8]* %v6, i32 0, i32 0
  call void @llvm.lifetime.start(i64 101, i8* %v12)
  call void @llvm.memset.p0i8.i32(i8* %v12, i8 0, i32 101, i32 8, i1 false)
  store i8 49, i8* %v12, align 8
  call void @test3(i8* %v7, i8* %v8, i8* %v9, i8* %v10, i8* %v11, i8* %v12)
  call void @llvm.lifetime.end(i64 101, i8* %v12)
  call void @llvm.lifetime.end(i64 100, i8* %v11)
  call void @llvm.lifetime.end(i64 4, i8* %v10)
  call void @llvm.lifetime.end(i64 4, i8* %v9)
  call void @llvm.lifetime.end(i64 2, i8* %v8)
  call void @llvm.lifetime.end(i64 1, i8* %v7)
  ret void
}

declare void @llvm.lifetime.start(i64, i8* nocapture) #0
declare void @llvm.lifetime.end(i64, i8* nocapture) #0
declare void @llvm.memset.p0i8.i32(i8* nocapture writeonly, i8, i32, i32, i1) #0

declare void @test3(i8*, i8*, i8*, i8*, i8*, i8*)
declare void @test4(i8*, i8*, i8*, i8*)

attributes #0 = { argmemonly nounwind "target-cpu"="hexagonv60" }
