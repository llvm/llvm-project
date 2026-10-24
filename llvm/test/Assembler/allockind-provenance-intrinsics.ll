; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; CHECK: declare void @f1(ptr) #0
declare void @f1(ptr) allockind("free,poisons_memory")

; CHECK: declare ptr @f2(i64) #1
declare ptr @f2(i64) allockind("alloc,uninitialized,address_unpredictable,alloc_disjoint")

; CHECK: declare noalias ptr @llvm.provenance.alloc.p0(ptr, i64) #2
declare ptr @llvm.provenance.alloc.p0(ptr, i64)

; CHECK: declare ptr @llvm.provenance.dealloc.p0(ptr allocptr captures(address)) #3
declare ptr @llvm.provenance.dealloc.p0(ptr)

define ptr @use(ptr %p) {
  %p.alloc = call ptr @llvm.provenance.alloc.p0(ptr %p, i64 8)
  %q = call ptr @llvm.provenance.dealloc.p0(ptr %p.alloc)
  ret ptr %q
}

; CHECK: attributes #0 = { allockind("free,poisons_memory") }
; CHECK: attributes #1 = { allockind("alloc,uninitialized,address_unpredictable,alloc_disjoint") }
; CHECK: attributes #2 = { nocallback nofree nosync nounwind willreturn allockind("alloc") allocsize(1) memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="provenance-alloc" }
; CHECK: attributes #3 = { nocallback nosync nounwind willreturn allockind("free") "alloc-family"="provenance-alloc" }
