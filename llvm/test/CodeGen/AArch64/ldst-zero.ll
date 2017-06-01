; RUN: llc -mtriple=aarch64 -mcpu=cortex-a53 < %s | FileCheck %s

; Tests to check that zero stores which are generated as STP xzr, xzr aren't
; scheduled incorrectly due to incorrect alias information

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1)
%struct.tree_common = type { i8*, i8*, i32 }

; Original test case which exhibited the bug
define void @test1(%struct.tree_common* %t, i32 %code, i8* %type) {
; CHECK-LABEL: test1:
; CHECK: stp xzr, xzr, [x0, #8]
; CHECK: stp xzr, x2, [x0]
; CHECK: str w1, [x0, #16]
entry:
  %0 = bitcast %struct.tree_common* %t to i8*
  tail call void @llvm.memset.p0i8.i64(i8* %0, i8 0, i64 24, i32 8, i1 false)
  %code1 = getelementptr inbounds %struct.tree_common, %struct.tree_common* %t, i64 0, i32 2
  store i32 %code, i32* %code1, align 8
  %type2 = getelementptr inbounds %struct.tree_common, %struct.tree_common* %t, i64 0, i32 1
  store i8* %type, i8** %type2, align 8
  ret void
}

; Store to each struct element instead of using memset
define void @test2(%struct.tree_common* %t, i32 %code, i8* %type) {
; CHECK-LABEL: test2:
; CHECK: stp xzr, xzr, [x0]
; CHECK: str wzr, [x0, #16]
; CHECK: str w1, [x0, #16]
; CHECK: str x2, [x0, #8]
entry:
  %0 = getelementptr inbounds %struct.tree_common, %struct.tree_common* %t, i64 0, i32 0
  %1 = getelementptr inbounds %struct.tree_common, %struct.tree_common* %t, i64 0, i32 1
  %2 = getelementptr inbounds %struct.tree_common, %struct.tree_common* %t, i64 0, i32 2
  store i8* zeroinitializer, i8** %0, align 8
  store i8* zeroinitializer, i8** %1, align 8
  store i32 zeroinitializer, i32* %2, align 8
  store i32 %code, i32* %2, align 8
  store i8* %type, i8** %1, align 8
  ret void
}

; Vector store instead of memset
define void @test3(%struct.tree_common* %t, i32 %code, i8* %type) {
; CHECK-LABEL: test3:
; CHECK: stp xzr, xzr, [x0, #8]
; CHECK: stp xzr, x2, [x0]
; CHECK: str w1, [x0, #16]
entry:
  %0 = bitcast %struct.tree_common* %t to <3 x i64>*
  store <3 x i64> zeroinitializer, <3 x i64>* %0, align 8
  %code1 = getelementptr inbounds %struct.tree_common, %struct.tree_common* %t, i64 0, i32 2
  store i32 %code, i32* %code1, align 8
  %type2 = getelementptr inbounds %struct.tree_common, %struct.tree_common* %t, i64 0, i32 1
  store i8* %type, i8** %type2, align 8
  ret void
}

; Vector store, then store to vector elements
define void @test4(<3 x i64>* %p, i64 %x, i64 %y) {
; CHECK-LABEL: test4:
; CHECK: stp xzr, xzr, [x0, #8]
; CHECK: stp xzr, x2, [x0]
; CHECK: str x1, [x0, #16]
entry:
  store <3 x i64> zeroinitializer, <3 x i64>* %p, align 8
  %0 = bitcast <3 x i64>* %p to i64*
  %1 = getelementptr inbounds i64, i64* %0, i64 2
  store i64 %x, i64* %1, align 8
  %2 = getelementptr inbounds i64, i64* %0, i64 1
  store i64 %y, i64* %2, align 8
  ret void
}
