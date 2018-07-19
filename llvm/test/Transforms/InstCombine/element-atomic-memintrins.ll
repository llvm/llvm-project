;; Placeholder tests that will fail once element atomic @llvm.mem[move|set] instrinsics have
;; been added to the MemIntrinsic class hierarchy. These will act as a reminder to
;; verify that inst combine handles these intrinsics properly once they have been
;; added to that class hierarchy.

; RUN: opt -instcombine -S < %s | FileCheck %s

;; ---- memset -----

; Ensure 0-length memset isn't removed
define void @test_memset_zero_length(i8* %dest) {
  ; CHECK-LABEL: test_memset_zero_length
  ; CHECK-NEXT: call void @llvm.memset.element.unordered.atomic.p0i8.i32(i8* align 1 %dest, i8 1, i32 0, i32 1)
  ; CHECK-NEXT: ret void
  call void @llvm.memset.element.unordered.atomic.p0i8.i32(i8* align 1 %dest, i8 1, i32 0, i32 1)
  ret void
}

; Ensure that small-sized memsets don't convert to stores
define void @test_memset_to_store(i8* %dest) {
  ; CHECK-LABEL: test_memset_to_store
  ; CHECK-NEXT: call void @llvm.memset.element.unordered.atomic.p0i8.i32(i8* align 1 %dest, i8 1, i32 1, i32 1)
  ; CHECK-NEXT: call void @llvm.memset.element.unordered.atomic.p0i8.i32(i8* align 1 %dest, i8 1, i32 2, i32 1)
  ; CHECK-NEXT: call void @llvm.memset.element.unordered.atomic.p0i8.i32(i8* align 1 %dest, i8 1, i32 4, i32 1)
  ; CHECK-NEXT: call void @llvm.memset.element.unordered.atomic.p0i8.i32(i8* align 1 %dest, i8 1, i32 8, i32 1)
  ; CHECK-NEXT: ret void
  call void @llvm.memset.element.unordered.atomic.p0i8.i32(i8* align 1 %dest, i8 1, i32 1, i32 1)
  call void @llvm.memset.element.unordered.atomic.p0i8.i32(i8* align 1 %dest, i8 1, i32 2, i32 1)
  call void @llvm.memset.element.unordered.atomic.p0i8.i32(i8* align 1 %dest, i8 1, i32 4, i32 1)
  call void @llvm.memset.element.unordered.atomic.p0i8.i32(i8* align 1 %dest, i8 1, i32 8, i32 1)
  ret void
}

declare void @llvm.memset.element.unordered.atomic.p0i8.i32(i8* nocapture writeonly, i8, i32, i32) nounwind argmemonly


;; =========================================
;; ----- memmove ------

; memmove from a global constant source does not become memcpy
@gconst = constant [8 x i8] c"0123456\00"
define void @test_memmove_to_memcpy(i8* %dest) {
  ; CHECK-LABEL: test_memmove_to_memcpy
  ; CHECK-NEXT: call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 1 %dest, i8* align 1 getelementptr inbounds ([8 x i8], [8 x i8]* @gconst, i64 0, i64 0), i32 8, i32 1)
  ; CHECK-NEXT: ret void
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 1 %dest, i8* align 1 getelementptr inbounds ([8 x i8], [8 x i8]* @gconst, i64 0, i64 0), i32 8, i32 1)
  ret void
}

define void @test_memmove_zero_length(i8* %dest, i8* %src) {
  ; CHECK-LABEL: test_memmove_zero_length
  ; CHECK-NEXT: call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 1 %dest, i8* align 1 %src, i32 0, i32 1)
  ; CHECK-NEXT: call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 2 %dest, i8* align 2 %src, i32 0, i32 2)
  ; CHECK-NEXT: call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 4 %dest, i8* align 4 %src, i32 0, i32 4)
  ; CHECK-NEXT: call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 8 %dest, i8* align 8 %src, i32 0, i32 8)
  ; CHECK-NEXT: call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 16 %dest, i8* align 16 %src, i32 0, i32 16)
  ; CHECK-NEXT: ret void
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 1 %dest, i8* align 1 %src, i32 0, i32 1)
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 2 %dest, i8* align 2 %src, i32 0, i32 2)
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 4 %dest, i8* align 4 %src, i32 0, i32 4)
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 8 %dest, i8* align 8 %src, i32 0, i32 8)
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 16 %dest, i8* align 16 %src, i32 0, i32 16)
  ret void  
}

; memmove with src==dest is removed
define void @test_memmove_removed(i8* %srcdest, i32 %sz) {
  ; CHECK-LABEL: test_memmove_removed
  ; CHECK-NEXT: call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 1 %srcdest, i8* align 1 %srcdest, i32 %sz, i32 1)
  ; CHECK-NEXT: call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 2 %srcdest, i8* align 2 %srcdest, i32 %sz, i32 2)
  ; CHECK-NEXT: call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 4 %srcdest, i8* align 4 %srcdest, i32 %sz, i32 4)
  ; CHECK-NEXT: call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 8 %srcdest, i8* align 8 %srcdest, i32 %sz, i32 8)
  ; CHECK-NEXT: call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 16 %srcdest, i8* align 16 %srcdest, i32 %sz, i32 16)
  ; CHECK-NEXT: ret void
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 1 %srcdest, i8* align 1 %srcdest, i32 %sz, i32 1)
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 2 %srcdest, i8* align 2 %srcdest, i32 %sz, i32 2)
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 4 %srcdest, i8* align 4 %srcdest, i32 %sz, i32 4)
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 8 %srcdest, i8* align 8 %srcdest, i32 %sz, i32 8)
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 16 %srcdest, i8* align 16 %srcdest, i32 %sz, i32 16)
  ret void
}

; memmove with a small constant length is converted to a load/store pair
define void @test_memmove_loadstore(i8* %dest, i8* %src) {
  ; CHECK-LABEL: test_memmove_loadstore
  ; CHECK-NEXT: call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 1 %dest, i8* align 1 %src, i32 1, i32 1)
  ; CHECK-NEXT: call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 1 %dest, i8* align 1 %src, i32 2, i32 1)
  ; CHECK-NEXT: call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 1 %dest, i8* align 1 %src, i32 4, i32 1)
  ; CHECK-NEXT: call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 1 %dest, i8* align 1 %src, i32 8, i32 1)
  ; CHECK-NEXT: ret void
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 1 %dest, i8* align 1 %src, i32 1, i32 1)
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 1 %dest, i8* align 1 %src, i32 2, i32 1)
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 1 %dest, i8* align 1 %src, i32 4, i32 1)
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 1 %dest, i8* align 1 %src, i32 8, i32 1)
  ret void
}

declare void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i32) nounwind argmemonly
