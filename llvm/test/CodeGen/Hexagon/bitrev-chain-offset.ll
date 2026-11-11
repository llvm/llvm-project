; RUN: llc -mtriple=hexagon -stop-after=hexagon-isel -o - %s | FileCheck %s

; A bit-reverse load accesses the base pointer with its low 16 bits reversed,
; and post-increments the base pointer by the modifier value. For a chain of
; bit-reverse loads the offset accessed by a load is therefore the bit-reverse
; of the sum of the modifiers of the preceding loads in the chain. Check that
; the offset of the memory operand is set accordingly, so that alias analysis
; does not break the dependence between an aliasing store and the loads.

@buf = global [256 x i8] zeroinitializer, align 65536

; CHECK-LABEL: name: {{.*}}brev_chain
; CHECK: L2_loadrb_pbr{{.*}}:: (load (s32) from @buf)
; CHECK: L2_loadrb_pbr{{.*}}:: (load (s32) from @buf + 128)
; CHECK: L2_loadrb_pbr{{.*}}:: (load (s32) from @buf + 64)
; CHECK: L2_loadrb_pbr{{.*}}:: (load (s32) from @buf + 192)
define i8 @brev_chain() {
entry:
  %v0 = tail call { i32, ptr } @llvm.hexagon.L2.loadrb.pbr(ptr @buf, i32 256)
  %p0 = extractvalue { i32, ptr } %v0, 1
  %v1 = tail call { i32, ptr } @llvm.hexagon.L2.loadrb.pbr(ptr %p0, i32 256)
  %p1 = extractvalue { i32, ptr } %v1, 1
  %v2 = tail call { i32, ptr } @llvm.hexagon.L2.loadrb.pbr(ptr %p1, i32 256)
  %p2 = extractvalue { i32, ptr } %v2, 1
  %v3 = tail call { i32, ptr } @llvm.hexagon.L2.loadrb.pbr(ptr %p2, i32 256)
  %r = extractvalue { i32, ptr } %v3, 0
  %t = trunc i32 %r to i8
  ret i8 %t
}

; The offset of the second load is not known, because the modifier of the
; first load is not a constant.

; CHECK-LABEL: name: {{.*}}brev_unknown_mod
; CHECK: L2_loadrb_pbr{{.*}}:: (load (s32) from @buf)
; CHECK: L2_loadrb_pbr{{.*}}:: (load (s32))
define i8 @brev_unknown_mod(i32 %m) {
entry:
  %v0 = tail call { i32, ptr } @llvm.hexagon.L2.loadrb.pbr(ptr @buf, i32 %m)
  %p0 = extractvalue { i32, ptr } %v0, 1
  %v1 = tail call { i32, ptr } @llvm.hexagon.L2.loadrb.pbr(ptr %p0, i32 256)
  %r = extractvalue { i32, ptr } %v1, 0
  %t = trunc i32 %r to i8
  ret i8 %t
}

; Only the low 16 bits of the base pointer take part in the bit-reverse, so
; the offset of the second load is not known if the sum of the modifiers does
; not fit in 16 unsigned bits.

; CHECK-LABEL: name: {{.*}}brev_wide_mod
; CHECK: L2_loadrb_pbr{{.*}}:: (load (s32) from @buf)
; CHECK: L2_loadrb_pbr{{.*}}:: (load (s32))
define i8 @brev_wide_mod() {
entry:
  %v0 = tail call { i32, ptr } @llvm.hexagon.L2.loadrb.pbr(ptr @buf, i32 65536)
  %p0 = extractvalue { i32, ptr } %v0, 1
  %v1 = tail call { i32, ptr } @llvm.hexagon.L2.loadrb.pbr(ptr %p0, i32 256)
  %r = extractvalue { i32, ptr } %v1, 0
  %t = trunc i32 %r to i8
  ret i8 %t
}

; A negative modifier makes the offset of the second load unknown as well.

; CHECK-LABEL: name: {{.*}}brev_negative_mod
; CHECK: L2_loadrb_pbr{{.*}}:: (load (s32) from @buf)
; CHECK: L2_loadrb_pbr{{.*}}:: (load (s32))
define i8 @brev_negative_mod() {
entry:
  %v0 = tail call { i32, ptr } @llvm.hexagon.L2.loadrb.pbr(ptr @buf, i32 -256)
  %p0 = extractvalue { i32, ptr } %v0, 1
  %v1 = tail call { i32, ptr } @llvm.hexagon.L2.loadrb.pbr(ptr %p0, i32 256)
  %r = extractvalue { i32, ptr } %v1, 0
  %t = trunc i32 %r to i8
  ret i8 %t
}

declare { i32, ptr } @llvm.hexagon.L2.loadrb.pbr(ptr, i32)
