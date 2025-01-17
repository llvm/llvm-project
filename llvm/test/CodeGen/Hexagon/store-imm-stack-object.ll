; RUN: llc -mtriple=hexagon < %s | FileCheck %s

target triple = "hexagon"

; CHECK-LABEL: test1:
; CHECK-DAG: memw(r29+#12) = ##875770417
; CHECK-DAG: memw(r29+#8) = #51
; CHECK-DAG: memh(r29+#6) = #50
; CHECK-DAG: memb(r29+#5) = #49
define void @test1() {
b0:
  %v1 = alloca [1 x i8], align 1
  %v2 = alloca i16, align 2
  %v3 = alloca i32, align 4
  %v4 = alloca i32, align 4
  call void @llvm.lifetime.start(i64 1, ptr %v1)
  store i8 49, ptr %v1, align 1
  call void @llvm.lifetime.start(i64 2, ptr %v2)
  store i16 50, ptr %v2, align 2
  call void @llvm.lifetime.start(i64 4, ptr %v3)
  store i32 51, ptr %v3, align 4
  call void @llvm.lifetime.start(i64 4, ptr %v4)
  store i32 875770417, ptr %v4, align 4
  call void @test4(ptr %v1, ptr %v2, ptr %v3, ptr %v4)
  call void @llvm.lifetime.end(i64 4, ptr %v4)
  call void @llvm.lifetime.end(i64 4, ptr %v3)
  call void @llvm.lifetime.end(i64 2, ptr %v2)
  call void @llvm.lifetime.end(i64 1, ptr %v1)
  ret void
}

; CHECK-LABEL: test2:
; CHECK-DAG: memw(r29+#8) = #51
; CHECK-DAG: memh(r29+#6) = r{{[0-9]+}}
; CHECK-DAG: memb(r29+#5) = r{{[0-9]+}}
define void @test2() {
b0:
  %v1 = alloca [1 x i8], align 1
  %v2 = alloca i16, align 2
  %v3 = alloca i32, align 4
  %v4 = alloca i32, align 4
  %v5 = alloca [100 x i8], align 8
  %v6 = alloca [101 x i8], align 8
  call void @llvm.lifetime.start(i64 1, ptr %v1)
  store i8 49, ptr %v1, align 1
  call void @llvm.lifetime.start(i64 2, ptr %v2)
  store i16 50, ptr %v2, align 2
  call void @llvm.lifetime.start(i64 4, ptr %v3)
  store i32 51, ptr %v3, align 4
  call void @llvm.lifetime.start(i64 4, ptr %v4)
  store i32 875770417, ptr %v4, align 4
  call void @llvm.lifetime.start(i64 100, ptr %v5)
  call void @llvm.memset.p0.i32(ptr align 8 %v5, i8 0, i32 100, i1 false)
  store i8 50, ptr %v5, align 8
  call void @llvm.lifetime.start(i64 101, ptr %v6)
  call void @llvm.memset.p0.i32(ptr align 8 %v6, i8 0, i32 101, i1 false)
  store i8 49, ptr %v6, align 8
  call void @test3(ptr %v1, ptr %v2, ptr %v3, ptr %v4, ptr %v5, ptr %v6)
  call void @llvm.lifetime.end(i64 101, ptr %v6)
  call void @llvm.lifetime.end(i64 100, ptr %v5)
  call void @llvm.lifetime.end(i64 4, ptr %v4)
  call void @llvm.lifetime.end(i64 4, ptr %v3)
  call void @llvm.lifetime.end(i64 2, ptr %v2)
  call void @llvm.lifetime.end(i64 1, ptr %v1)
  ret void
}

declare void @llvm.lifetime.start(i64, ptr nocapture) #0
declare void @llvm.lifetime.end(i64, ptr nocapture) #0
declare void @llvm.memset.p0.i32(ptr nocapture writeonly, i8, i32, i1) #0

declare void @test3(ptr, ptr, ptr, ptr, ptr, ptr)
declare void @test4(ptr, ptr, ptr, ptr)

attributes #0 = { argmemonly nounwind "target-cpu"="hexagonv60" }
