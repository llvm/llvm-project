; RUN: llc -O3 -ppc-late-peephole=false -o - %s | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

%class1 = type { %union1 }
%union1 = type { i64, [24 x i8] }
%class2 = type { %class3 }
%class3 = type { %class4 }
%class4 = type { %class5, i64, %union.anon }
%class5 = type { i8* }
%union.anon = type { i64, [8 x i8] }

@ext = external global %"class1", align 8

; We can't select lxv for this because even though we're accessing an offset of
; 16 from the stack slot, the stack slot is only guaranteed to be 8-byte
; aligned. When the frame is finalized it is converted to lxv (frame-reg) +
; (offset + 16). Because offset isn't guaranteed to be 16-byte aligned, we may
; end up needing to translate the lxv instruction to lxvx
; CHECK-LABEL: unaligned_slot:
; CHECK-NOT: lxv {{[0-9]+}}, {{[-0-9]+}}({{[0-9]+}})
; CHECK: blr
define void @unaligned_slot() #0 {
  %1 = alloca %class2, align 8
  %2 = getelementptr inbounds %class2, %class2* %1, i64 0, i32 0, i32 0, i32 2
  %3 = bitcast %union.anon* %2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 nonnull getelementptr inbounds (%class1, %class1* @ext, i64 0, i32 0, i32 1, i64 8), i8* align 8 nonnull %3, i64 16, i1 false) #2
  ret void
}
; CHECK-LABEL: aligned_slot:
; CHECK: lxv {{[0-9]+}}, {{[-0-9]+}}({{[0-9]+}})
; CHECK: blr
define void @aligned_slot() #0 {
  %1 = alloca %class2, align 16
  %2 = getelementptr inbounds %class2, %class2* %1, i64 0, i32 0, i32 0, i32 2
  %3 = bitcast %union.anon* %2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 nonnull getelementptr inbounds (%class1, %class1* @ext, i64 0, i32 0, i32 1, i64 8), i8* align 8 nonnull %3, i64 16, i1 false) #2
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

attributes #0 = { nounwind "target-cpu"="pwr9" "target-features"="+altivec,+bpermd,+crypto,+direct-move,+extdiv,+htm,+power8-vector,+power9-vector,+vsx,-qpx" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind }
