; RUN: opt -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 %s | FileCheck %s

declare void @llvm.memset.element.unordered.atomic.p0.i32(ptr, i8, i64, i32)

define void @test_memset_element_unordered_atomic_const_size(ptr noalias %a) {
; CHECK-LABEL: Function: test_memset_element_unordered_atomic_const_size
; CHECK:       Just Mod:  Ptr: i8* %a	<->  call void @llvm.memset.element.unordered.atomic.p0.i64(ptr align 1 %a, i8 0, i64 4, i32 1)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %a.gep.1	<->  call void @llvm.memset.element.unordered.atomic.p0.i64(ptr align 1 %a, i8 0, i64 4, i32 1)
; CHECK-NEXT:  NoModRef:  Ptr: i8* %a.gep.5	<->  call void @llvm.memset.element.unordered.atomic.p0.i64(ptr align 1 %a, i8 0, i64 4, i32 1)
;
entry:
  load i8, ptr %a
  call void @llvm.memset.element.unordered.atomic.p0.i32(ptr align 1 %a, i8 0, i64 4, i32 1)
  %a.gep.1 = getelementptr i8, ptr %a, i32 1
  store i8 0, ptr %a.gep.1
  %a.gep.5 = getelementptr i8, ptr %a, i32 5
  store i8 1, ptr %a.gep.5
  ret void
}

define void @test_memset_element_unordered_atomic_variable_size(ptr noalias %a, i64 %n) {
; CHECK-LABEL: Function: test_memset_element_unordered_atomic_variable_size
; CHECK:       Just Mod:  Ptr: i8* %a	<->  call void @llvm.memset.element.unordered.atomic.p0.i64(ptr align 1 %a, i8 0, i64 %n, i32 1)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %a.gep.1	<->  call void @llvm.memset.element.unordered.atomic.p0.i64(ptr align 1 %a, i8 0, i64 %n, i32 1)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %a.gep.5	<->  call void @llvm.memset.element.unordered.atomic.p0.i64(ptr align 1 %a, i8 0, i64 %n, i32 1)
;
entry:
  load i8, ptr %a
  call void @llvm.memset.element.unordered.atomic.p0.i32(ptr align 1 %a, i8 0, i64 %n, i32 1)
  %a.gep.1 = getelementptr i8, ptr %a, i32 1
  store i8 0, ptr %a.gep.1
  %a.gep.5 = getelementptr i8, ptr %a, i32 5
  store i8 1, ptr %a.gep.5
  ret void
}

declare void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i32)

define void @test_memcpy_element_unordered_atomic_const_size(ptr noalias %a, ptr noalias %b) {
; CHECK-LABEL: Function: test_memcpy_element_unordered_atomic_const_size
; CHECK:       Just Ref:  Ptr: i8* %a	<->  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 4, i32 1)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %b	<->  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 4, i32 1)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %a.gep.1	<->  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 4, i32 1)
; CHECK-NEXT:  NoModRef:  Ptr: i8* %a.gep.5	<->  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 4, i32 1)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %b.gep.1	<->  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 4, i32 1)
; CHECK-NEXT:  NoModRef:  Ptr: i8* %b.gep.5	<->  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 4, i32 1)
;
entry:
  load i8, ptr %a
  load i8, ptr %b
  %a.gep.1 = getelementptr i8, ptr %a, i32 1
  store i8 0, ptr %a.gep.1
  %a.gep.5 = getelementptr i8, ptr %a, i32 5
  store i8 1, ptr %a.gep.5
  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 4, i32 1)
  %b.gep.1 = getelementptr i8, ptr %b, i32 1
  store i8 0, ptr %b.gep.1
  %b.gep.5 = getelementptr i8, ptr %b, i32 5
  store i8 1, ptr %b.gep.5
  ret void
}

define void @test_memcpy_element_unordered_atomic_variable_size(ptr noalias %a, ptr noalias %b, i64 %n) {
; CHECK-LABEL: Function: test_memcpy_element_unordered_atomic_variable_size
; CHECK:       Just Ref:  Ptr: i8* %a	<->  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 %n, i32 1)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %b	<->  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 %n, i32 1)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %a.gep.1	<->  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 %n, i32 1)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %a.gep.5	<->  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 %n, i32 1)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %b.gep.1	<->  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 %n, i32 1)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %b.gep.5	<->  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 %n, i32 1)
;
entry:
  load i8, ptr %a
  load i8, ptr %b
  %a.gep.1 = getelementptr i8, ptr %a, i32 1
  store i8 0, ptr %a.gep.1
  %a.gep.5 = getelementptr i8, ptr %a, i32 5
  store i8 1, ptr %a.gep.5
  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 %n, i32 1)
  %b.gep.1 = getelementptr i8, ptr %b, i32 1
  store i8 0, ptr %b.gep.1
  %b.gep.5 = getelementptr i8, ptr %b, i32 5
  store i8 1, ptr %b.gep.5
  ret void
}

declare void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i32)

define void @test_memmove_element_unordered_atomic_const_size(ptr noalias %a, ptr noalias %b) {
; CHECK-LABEL: Function: test_memmove_element_unordered_atomic_const_size
; CHECK:       Just Ref:  Ptr: i8* %a	<->  call void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 4, i32 1)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %b	<->  call void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 4, i32 1)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %a.gep.1	<->  call void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 4, i32 1)
; CHECK-NEXT:  NoModRef:  Ptr: i8* %a.gep.5	<->  call void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 4, i32 1)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %b.gep.1	<->  call void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 4, i32 1)
; CHECK-NEXT:  NoModRef:  Ptr: i8* %b.gep.5	<->  call void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 4, i32 1)
;
entry:
  load i8, ptr %a
  load i8, ptr %b
  %a.gep.1 = getelementptr i8, ptr %a, i32 1
  store i8 0, ptr %a.gep.1
  %a.gep.5 = getelementptr i8, ptr %a, i32 5
  store i8 1, ptr %a.gep.5
  call void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 4, i32 1)
  %b.gep.1 = getelementptr i8, ptr %b, i32 1
  store i8 0, ptr %b.gep.1
  %b.gep.5 = getelementptr i8, ptr %b, i32 5
  store i8 1, ptr %b.gep.5
  ret void
}

define void @test_memmove_element_unordered_atomic_variable_size(ptr noalias %a, ptr noalias %b, i64 %n) {
; CHECK-LABEL: Function: test_memmove_element_unordered_atomic_variable_size
; CHECK:       Just Ref:  Ptr: i8* %a	<->  call void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 %n, i32 1)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %b	<->  call void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 %n, i32 1)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %a.gep.1	<->  call void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 %n, i32 1)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %a.gep.5	<->  call void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 %n, i32 1)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %b.gep.1	<->  call void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 %n, i32 1)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %b.gep.5	<->  call void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 %n, i32 1)
;
entry:
  load i8, ptr %a
  load i8, ptr %b
  %a.gep.1 = getelementptr i8, ptr %a, i32 1
  store i8 0, ptr %a.gep.1
  %a.gep.5 = getelementptr i8, ptr %a, i32 5
  store i8 1, ptr %a.gep.5
  call void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 %n, i32 1)
  %b.gep.1 = getelementptr i8, ptr %b, i32 1
  store i8 0, ptr %b.gep.1
  %b.gep.5 = getelementptr i8, ptr %b, i32 5
  store i8 1, ptr %b.gep.5
  ret void
}
