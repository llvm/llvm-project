; RUN: llc -mtriple=hexagon -stop-after=hexagon-isel -o - %s | FileCheck %s

; A bit-reverse load accesses the base pointer with its low 16 bits reversed,
; so the accessed address is only known to stay inside the underlying object
; when that object is aligned to 64K. Check that no pointer information is
; attached to the memory operand otherwise, so that alias analysis stays
; conservative and an aliasing store is not moved across the loads.

@aligned64k = global [256 x i8] zeroinitializer, align 65536
@aligned256 = global [256 x i8] zeroinitializer, align 256

; CHECK-LABEL: name: {{.*}}brev_aligned64k
; CHECK: L2_loadrb_pbr{{.*}}:: (load (s32) from @aligned64k)
; CHECK: L2_loadrb_pbr{{.*}}:: (load (s32) from @aligned64k + 128)
define i8 @brev_aligned64k() {
entry:
  %v0 = tail call { i32, ptr } @llvm.hexagon.L2.loadrb.pbr(ptr @aligned64k, i32 256)
  %p0 = extractvalue { i32, ptr } %v0, 1
  %v1 = tail call { i32, ptr } @llvm.hexagon.L2.loadrb.pbr(ptr %p0, i32 256)
  %r = extractvalue { i32, ptr } %v1, 0
  %t = trunc i32 %r to i8
  ret i8 %t
}

; CHECK-LABEL: name: {{.*}}brev_aligned256
; CHECK: L2_loadrb_pbr{{.*}}:: (load (s32))
; CHECK: L2_loadrb_pbr{{.*}}:: (load (s32))
define i8 @brev_aligned256() {
entry:
  %v0 = tail call { i32, ptr } @llvm.hexagon.L2.loadrb.pbr(ptr @aligned256, i32 256)
  %p0 = extractvalue { i32, ptr } %v0, 1
  %v1 = tail call { i32, ptr } @llvm.hexagon.L2.loadrb.pbr(ptr %p0, i32 256)
  %r = extractvalue { i32, ptr } %v1, 0
  %t = trunc i32 %r to i8
  ret i8 %t
}

; The alignment of an incoming pointer is not known.

; CHECK-LABEL: name: {{.*}}brev_unknown_align
; CHECK: L2_loadrb_pbr{{.*}}:: (load (s32))
define i8 @brev_unknown_align(ptr %p) {
entry:
  %v0 = tail call { i32, ptr } @llvm.hexagon.L2.loadrb.pbr(ptr %p, i32 256)
  %r = extractvalue { i32, ptr } %v0, 0
  %t = trunc i32 %r to i8
  ret i8 %t
}

declare { i32, ptr } @llvm.hexagon.L2.loadrb.pbr(ptr, i32)
