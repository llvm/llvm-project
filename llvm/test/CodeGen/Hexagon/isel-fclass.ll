; Tests lowering of sfclass/dfclass compares.
; Sub-optimal code
;         {
;                p0 = sfclass(r0,#16)
;                r0 = sfadd(r0,r0)
;        }
;        {
;                r2 = p0
;        }
;        {
;                if (p0.new) r0 = ##1065353216
;                p0 = cmp.eq(r2,#0)
;                jumpr r31
;        }
; With the patterns added, we should be generating
;        {
;                p0 = sfclass(r0,#16)
;                r0 = sfadd(r0,r0)
;        }
;        {
;                if (!p0) r0 = ##1065353216
;                jumpr r31
;        }

; RUN: llc -march=hexagon -stop-after=hexagon-isel %s -o - | FileCheck %s

; CHECK: bb.0.entry1
; CHECK: F2_sfclass
; CHECK-NOT: C2_cmp
; CHECK: C2_not
; CHECK: F2_sfadd
; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define float @test1(float noundef %x) {
entry1:
  %0 = tail call i32 @llvm.hexagon.F2.sfclass(float %x, i32 16)
  %tobool.not = icmp eq i32 %0, 0
  %add = fadd float %x, %x
  %spec.select = select i1 %tobool.not, float 1.000000e+00, float %add
  ret float %spec.select
}

; CHECK: bb.0.entry2
; CHECK: F2_sfclass
; CHECK-NOT: C2_cmp
; CHECK: F2_sfadd
define float @test2(float noundef %x) {
entry2:
  %0 = tail call i32 @llvm.hexagon.F2.sfclass(float %x, i32 16)
  %tobool.not = icmp eq i32 %0, 0
  %add = fadd float %x, %x
  %spec.select = select i1 %tobool.not, float %add, float 1.000000e+00
  ret float %spec.select
}

; CHECK: bb.0.entry3
; CHECK: F2_dfclass
; CHECK-NOT: C2_cmp
; CHECK: C2_not
; CHECK: F2_dfadd
define double @test3(double noundef %x) {
entry3:
  %0 = tail call i32 @llvm.hexagon.F2.dfclass(double %x, i32 16)
  %tobool.not = icmp eq i32 %0, 0
  %add = fadd double %x, %x
  %spec.select = select i1 %tobool.not, double 1.000000e+00, double %add
  ret double %spec.select
}

; CHECK: bb.0.entry4
; CHECK: F2_dfclass
; CHECK-NOT: C2_cmp
; CHECK: F2_dfadd
define double @test4(double noundef %x) {
entry4:
  %0 = tail call i32 @llvm.hexagon.F2.dfclass(double %x, i32 16)
  %tobool.not = icmp eq i32 %0, 0
  %add = fadd double %x, %x
  %spec.select = select i1 %tobool.not, double %add, double 1.000000e+00
  ret double %spec.select
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.hexagon.F2.dfclass(double, i32 immarg)

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.hexagon.F2.sfclass(float, i32 immarg)
