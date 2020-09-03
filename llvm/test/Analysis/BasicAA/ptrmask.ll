; RUN: opt -basic-aa -aa-eval -print-no-aliases -disable-output %s 2>&1 | FileCheck %s

%struct = type <{ [20 x i64] }>

; CHECK-LABEL: Function: test_noalias: 4 pointers, 1 call sites
; CHECK-NEXT:  NoAlias:	%struct* %ptr1, i64* %ptr2
; CHECK-NEXT:  NoAlias:	%struct* %addr.ptr, i64* %ptr2
; CHECK-NEXT:  NoAlias:	i64* %gep, i64* %ptr2
define void @test_noalias(%struct* noalias %ptr1, i64* %ptr2, i64 %offset) {
entry:
  %addr.ptr = call %struct* @llvm.ptrmask.p0s_struct.p0s.struct.i64(%struct* %ptr1, i64 72057594037927928)
  store i64 10, i64* %ptr2
  %gep = getelementptr inbounds %struct, %struct* %addr.ptr, i64 0, i32 0, i64 %offset
  store i64 1, i64* %gep, align 8
  ret void
}

; CHECK-NEXT: Function: test_alias: 4 pointers, 1 call sites
; CHECK-NOT: NoAlias
define void @test_alias(%struct* %ptr1, i64* %ptr2, i64 %offset) {
entry:
  %addr.ptr = call %struct* @llvm.ptrmask.p0s_struct.p0s.struct.i64(%struct* %ptr1, i64 72057594037927928)
  store i64 10, i64* %ptr2
  %gep = getelementptr inbounds %struct, %struct* %addr.ptr, i64 0, i32 0, i64 %offset
  store i64 1, i64* %gep, align 8
  ret void
}

declare %struct* @llvm.ptrmask.p0s_struct.p0s.struct.i64(%struct*, i64)
