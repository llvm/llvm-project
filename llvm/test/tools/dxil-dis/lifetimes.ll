; RUN: llc --filetype=obj %s -o - | dxil-dis -o - | FileCheck %s
target triple = "dxil-unknown-shadermodel6.7-library"

define void @test_lifetimes()  {
; CHECK-LABEL: test_lifetimes
; CHECK-NEXT: [[ALLOCA:%.*]] = alloca [2 x i32], align 4
; CHECK-NEXT: [[GEP:%.*]] = getelementptr [2 x i32], [2 x i32]* [[ALLOCA]], i32 0, i32 0
; CHECK-NEXT: [[BITCAST:%.*]] = bitcast [2 x i32]* [[ALLOCA]] to i8*
; CHECK-NEXT: call void @llvm.lifetime.start(i64 8, i8* nonnull [[BITCAST]])
; CHECK-NEXT: store i32 0, i32* [[GEP]], align 4
; CHECK-NEXT: [[BITCAST:%.*]] = bitcast [2 x i32]* [[ALLOCA]] to i8*
; CHECK-NEXT: call void @llvm.lifetime.end(i64 8, i8* nonnull [[BITCAST]])
; CHECK-NEXT: ret void
;
  %a = alloca [2 x i32], align 4
  %gep = getelementptr [2 x i32], ptr %a, i32 0, i32 0
  call void @llvm.lifetime.start.p0(ptr nonnull %a)
  store i32 0, ptr %gep, align 4
  call void @llvm.lifetime.end.p0(ptr nonnull %a)
  ret void
}

; CHECK-DAG: attributes [[LIFETIME_ATTRS:#.*]] = { nounwind }

; CHECK-DAG: ; Function Attrs: nounwind
; CHECK-DAG: declare void @llvm.lifetime.start(i64, i8* nocapture) [[LIFETIME_ATTRS]]

; CHECK-DAG: ; Function Attrs: nounwind
; CHECK-DAG: declare void @llvm.lifetime.end(i64, i8* nocapture) [[LIFETIME_ATTRS]]

; Function Attrs: nounwind memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr) #0

; Function Attrs: nounwind memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr) #0

attributes #0 = { nounwind memory(argmem: readwrite) }

