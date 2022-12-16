; RUN: opt < %s -passes=lint -disable-output 2>&1 | FileCheck %s

%s = type { i8 }

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0.p0.i32(ptr nocapture writeonly, ptr nocapture readonly, i32, i1) #0

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0.i32(ptr nocapture writeonly, i8, i32, i1) #0

declare void @f1(ptr noalias nocapture sret(%s), ptr nocapture readnone)

define void @f2() {
entry:
  %c = alloca %s
  %tmp = alloca %s
  call void @llvm.memset.p0.i32(ptr %c, i8 0, i32 1, i1 false)
  call void @f1(ptr sret(%s) %c, ptr %c)
  ret void
}

; Lint should complain about us passing %c to both arguments since one of them
; is noalias.
; CHECK: Unusual: noalias argument aliases another argument
; CHECK-NEXT: call void @f1(ptr sret(%s) %c, ptr %c)

declare void @f3(ptr noalias nocapture sret(%s), ptr byval(%s) nocapture readnone)

define void @f4() {
entry:
  %c = alloca %s
  %tmp = alloca %s
  call void @llvm.memset.p0.i32(ptr %c, i8 0, i32 1, i1 false)
  call void @f3(ptr sret(%s) %c, ptr byval(%s) %c)
  ret void
}

; Lint should not complain about passing %c to both arguments even if one is
; noalias, since the other one is byval, effectively copying the data to the
; stack instead of passing the pointer itself.
; CHECK-NOT: Unusual: noalias argument aliases another argument
; CHECK-NOT: call void @f3(ptr sret(%s) %c, ptr byval(%s) %c)

attributes #0 = { argmemonly nounwind }
