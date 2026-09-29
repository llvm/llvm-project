; RUN: opt -aa-pipeline=basic-aa -passes=aa-eval -print-no-aliases -disable-output %s 2>&1 | FileCheck %s

%struct = type <{ [20 x i64] }>

; CHECK-LABEL: Function: test_noalias
; CHECK-NEXT:  NoAlias:	%struct* %ptr1, i64* %ptr2
; CHECK-NEXT:  NoAlias:	%struct* %addr.ptr, i64* %ptr2
; CHECK-NEXT:  NoAlias:	i64* %gep, i64* %ptr2
define void @test_noalias(ptr noalias %ptr1, ptr %ptr2, i64 %offset) {
entry:
  %addr.ptr = call ptr @llvm.ptrmask.p0.p0.struct.i64(ptr %ptr1, i64 72057594037927928)
  load %struct, ptr %ptr1
  load %struct, ptr %addr.ptr
  store i64 10, ptr %ptr2
  %gep = getelementptr inbounds %struct, ptr %addr.ptr, i64 0, i32 0, i64 %offset
  store i64 1, ptr %gep, align 8
  ret void
}

; CHECK-NEXT: Function: test_alias
; CHECK-NOT: NoAlias
define void @test_alias(ptr %ptr1, ptr %ptr2, i64 %offset) {
entry:
  %addr.ptr = call ptr @llvm.ptrmask.p0.p0.struct.i64(ptr %ptr1, i64 72057594037927928)
  load %struct, ptr %ptr1
  load %struct, ptr %addr.ptr
  store i64 10, ptr %ptr2
  %gep = getelementptr inbounds %struct, ptr %addr.ptr, i64 0, i32 0, i64 %offset
  store i64 1, ptr %gep, align 8
  ret void
}

declare ptr @llvm.ptrmask.p0.p0.struct.i64(ptr, i64)
