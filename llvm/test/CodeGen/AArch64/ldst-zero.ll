; RUN: llc -mtriple=aarch64 -mcpu=cortex-a53 < %s | FileCheck %s

; Tests to check that zero stores which are generated as STP xzr, xzr aren't
; scheduled incorrectly due to incorrect alias information

declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1)
%struct.tree_common = type { ptr, ptr, i32 }

; Original test case which exhibited the bug
define void @test1(ptr %t, i32 %code, ptr %type) {
; CHECK-LABEL: test1:
; CHECK-DAG: stp x2, xzr, [x0, #8]
; CHECK-DAG: str w1, [x0, #16]
; CHECK-DAG: str xzr, [x0]
entry:
  tail call void @llvm.memset.p0.i64(ptr align 8 %t, i8 0, i64 24, i1 false)
  %code1 = getelementptr inbounds %struct.tree_common, ptr %t, i64 0, i32 2
  store i32 %code, ptr %code1, align 8
  %type2 = getelementptr inbounds %struct.tree_common, ptr %t, i64 0, i32 1
  store ptr %type, ptr %type2, align 8
  ret void
}

; Store to each struct element instead of using memset
define void @test2(ptr %t, i32 %code, ptr %type) {
; CHECK-LABEL: test2:
; CHECK-DAG: str w1, [x0, #16]
; CHECK-DAG: stp xzr, x2, [x0]
entry:
  %0 = getelementptr inbounds %struct.tree_common, ptr %t, i64 0, i32 1
  %1 = getelementptr inbounds %struct.tree_common, ptr %t, i64 0, i32 2
  store ptr zeroinitializer, ptr %t, align 8
  store ptr zeroinitializer, ptr %0, align 8
  store i32 zeroinitializer, ptr %1, align 8
  store i32 %code, ptr %1, align 8
  store ptr %type, ptr %0, align 8
  ret void
}

; Vector store instead of memset
define void @test3(ptr %t, i32 %code, ptr %type) {
; CHECK-LABEL: test3:
; CHECK-DAG: stp x2, xzr, [x0, #8]
; CHECK-DAG: str w1, [x0, #16]
; CHECK-DAG: str xzr, [x0]
entry:
  store <3 x i64> zeroinitializer, ptr %t, align 8
  %code1 = getelementptr inbounds %struct.tree_common, ptr %t, i64 0, i32 2
  store i32 %code, ptr %code1, align 8
  %type2 = getelementptr inbounds %struct.tree_common, ptr %t, i64 0, i32 1
  store ptr %type, ptr %type2, align 8
  ret void
}

; Vector store, then store to vector elements
define void @test4(ptr %p, i64 %x, i64 %y) {
; CHECK-LABEL: test4:
; CHECK-DAG: stp x2, x1, [x0, #8]
; CHECK-DAG: str xzr, [x0]
entry:
  store <3 x i64> zeroinitializer, ptr %p, align 8
  %0 = getelementptr inbounds i64, ptr %p, i64 2
  store i64 %x, ptr %0, align 8
  %1 = getelementptr inbounds i64, ptr %p, i64 1
  store i64 %y, ptr %1, align 8
  ret void
}
