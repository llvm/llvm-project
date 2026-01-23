; RUN: opt -mtriple=x86_64 -aa-pipeline=basic-aa -passes=inferattrs,aa-eval -print-all-alias-modref-info -disable-output 2>&1 %s | FileCheck %s

define void @test_memset_pattern4_const_size(ptr noalias %a, i32 %pattern) {
; CHECK-LABEL: Function: test_memset_pattern4_const_size
; CHECK:      Just Mod:  Ptr: i8* %a	<->  call void @llvm.experimental.memset.pattern.p0.i32.i64(ptr %a, i32 %pattern, i64 17, i1 false)
; CHECK-NEXT: Just Mod:  Ptr: i8* %a.gep.1	<->  call void @llvm.experimental.memset.pattern.p0.i32.i64(ptr %a, i32 %pattern, i64 17, i1 false)
; CHECK-NEXT: NoModRef:  Ptr: i8* %a.gep.129	<->  call void @llvm.experimental.memset.pattern.p0.i32.i64(ptr %a, i32 %pattern, i64 17, i1 false)

entry:
  load i8, ptr %a
  call void @llvm.experimental.memset.pattern(ptr %a, i32 %pattern, i64 17, i1 0)
  %a.gep.1 = getelementptr i8, ptr %a, i32 1
  store i8 0, ptr %a.gep.1
  %a.gep.129 = getelementptr i8, ptr %a, i32 129
  store i8 1, ptr %a.gep.129

  ret void
}
