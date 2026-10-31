; RUN: llc < %s -mtriple=x86_64_lfi -filetype=obj -o %t.o
; RUN: llvm-objdump -d --no-show-raw-insn %t.o | FileCheck %s

define void @copy(ptr %a, ptr %b) {
; CHECK-LABEL: <copy>:
; CHECK:      movl %edi, %edi
; CHECK-NEXT: leaq (%r14,%rdi), %rdi
; CHECK-NEXT: movl %esi, %esi
; CHECK-NEXT: leaq (%r14,%rsi), %rsi
; CHECK-NEXT: rep {{.*}}movsb
  call void @llvm.memcpy.inline.p0.p0.i64(ptr %a, ptr %b, i64 262144, i1 false)
  ret void
}

define void @fill(ptr %a, i8 %v) minsize {
; CHECK-LABEL: <fill>:
; CHECK:      movl %edi, %edi
; CHECK-NEXT: leaq (%r14,%rdi), %rdi
; CHECK-NEXT: rep {{.*}}stosb
  call void @llvm.memset.inline.p0.i64(ptr %a, i8 %v, i64 262144, i1 false)
  ret void
}

declare void @llvm.memcpy.inline.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memset.inline.p0.i64(ptr, i8, i64, i1)
