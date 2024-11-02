; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=opcodes --test FileCheck --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -check-prefixes=CHECK,RESULT %s < %t

; CHECK-LABEL: @fdiv_fast(
; RESULT-NEXT: %op = fmul fast float %a, %b, !dbg !7, !fpmath !13
; RESULT-NEXT: ret float %op
define float @fdiv_fast(float %a, float %b) {
  %op = fdiv fast float %a, %b, !dbg !7, !fpmath !13
  ret float %op
}

; CHECK-LABEL: @frem_nnan(
; RESULT-NEXT: %op = fmul nnan float %a, %b, !dbg !7, !fpmath !13
; RESULT-NEXT: ret float %op
define float @frem_nnan(float %a, float %b) {
  %op = frem nnan float %a, %b, !dbg !7, !fpmath !13
  ret float %op
}

; CHECK-LABEL: @udiv(
; RESULT-NEXT: %op = mul i32 %a, %b, !dbg !7
; RESULT-NEXT: ret i32 %op
define i32 @udiv(i32 %a, i32 %b) {
  %op = udiv i32 %a, %b, !dbg !7
  ret i32 %op
}

; CHECK-LABEL: @udiv_vec(
; RESULT-NEXT: %op = mul <2 x i32> %a, %b, !dbg !7
; RESULT-NEXT: ret <2 x i32> %op
define <2 x i32> @udiv_vec(<2 x i32> %a, <2 x i32> %b) {
  %op = udiv <2 x i32> %a, %b, !dbg !7
  ret <2 x i32> %op
}

; CHECK-LABEL: @sdiv(
; RESULT-NEXT: %op = mul i32 %a, %b{{$}}
; RESULT-NEXT: ret i32 %op
define i32 @sdiv(i32 %a, i32 %b) {
  %op = sdiv i32 %a, %b
  ret i32 %op
}

; CHECK-LABEL: @sdiv_exact(
; RESULT-NEXT: %op = mul i32 %a, %b, !dbg !7
; RESULT-NEXT: ret
define i32 @sdiv_exact(i32 %a, i32 %b) {
  %op = sdiv exact i32 %a, %b, !dbg !7
  ret i32 %op
}

; CHECK-LABEL: @urem(
; RESULT-NEXT: %op = mul i32 %a, %b, !dbg !7
; RESULT-NEXT: ret
define i32 @urem(i32 %a, i32 %b) {
  %op = urem i32 %a, %b, !dbg !7
  ret i32 %op
}

; CHECK-LABEL: @srem(
; RESULT-NEXT: %op = mul i32 %a, %b, !dbg !7
; RESULT-NEXT: ret
define i32 @srem(i32 %a, i32 %b) {
  %op = srem i32 %a, %b, !dbg !7
  ret i32 %op
}

; Make sure there's no crash if the IRBuilder decided to constant fold something
; CHECK-LABEL: @add_constant_fold(
; RESULT-NEXT: %op = add i32 0, 0, !dbg !7
; RESULT-NEXT: ret
define i32 @add_constant_fold() {
  %op = add i32 0, 0, !dbg !7
  ret i32 %op
}

; CHECK-LABEL: @add(
; RESULT-NEXT: %op = or i32 %a, %b, !dbg !7
; RESULT-NEXT: ret
define i32 @add(i32 %a, i32 %b) {
  %op = add i32 %a, %b, !dbg !7
  ret i32 %op
}

; CHECK-LABEL: @add_nuw(
; RESULT-NEXT: %op = or i32 %a, %b, !dbg !7
; RESULT-NEXT: ret
define i32 @add_nuw(i32 %a, i32 %b) {
  %op = add nuw i32 %a, %b, !dbg !7
  ret i32 %op
}

; CHECK-LABEL: @add_nsw(
; RESULT-NEXT: %op = or i32 %a, %b, !dbg !7
; RESULT-NEXT: ret
define i32 @add_nsw(i32 %a, i32 %b) {
  %op = add nsw i32 %a, %b, !dbg !7
  ret i32 %op
}

; CHECK-LABEL: @sub_nuw_nsw(
; RESULT-NEXT: %op = or i32 %a, %b, !dbg !7
; RESULT-NEXT: ret
define i32 @sub_nuw_nsw(i32 %a, i32 %b) {
  %op = sub nuw nsw i32 %a, %b, !dbg !7
  ret i32 %op
}

; CHECK-LABEL: @workitem_id_y(
; RESULT-NEXT: %id = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !7
; RESULT-NEXT: ret
define i32 @workitem_id_y() {
  %id = call i32 @llvm.amdgcn.workitem.id.y(), !dbg !7
  ret i32 %id
}

; CHECK-LABEL: @workitem_id_z(
; RESULT-NEXT: %id = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !7
; RESULT-NEXT: ret
define i32 @workitem_id_z() {
  %id = call i32 @llvm.amdgcn.workitem.id.z(), !dbg !7
  ret i32 %id
}

; CHECK-LABEL: @workgroup_id_y(
; RESULT-NEXT: %id = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !7
; RESULT-NEXT: ret
define i32 @workgroup_id_y() {
  %id = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !7
  ret i32 %id
}

; CHECK-LABEL: @workgroup_id_z(
; RESULT-NEXT: %id = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !7
; RESULT-NEXT: ret
define i32 @workgroup_id_z() {
  %id = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !7
  ret i32 %id
}

; CHECK-LABEL: @minnum_nsz(
; RESULT-NEXT: %op = fmul nsz float %a, %b, !dbg !7
; RESULT-NEXT: ret
define float @minnum_nsz(float %a, float %b) {
  %op = call nsz float @llvm.minnum.f32(float %a, float %b), !dbg !7
  ret float %op
}

; CHECK-LABEL: @maxnum_nsz(
; RESULT-NEXT: %op = fmul nsz float %a, %b, !dbg !7
; RESULT-NEXT: ret
define float @maxnum_nsz(float %a, float %b) {
  %op = call nsz float @llvm.maxnum.f32(float %a, float %b), !dbg !7
  ret float %op
}

; CHECK-LABEL: @minimum_nsz(
; RESULT-NEXT: %op = fmul nsz float %a, %b, !dbg !7
; RESULT-NEXT: ret
define float @minimum_nsz(float %a, float %b) {
  %op = call nsz float @llvm.minimum.f32(float %a, float %b), !dbg !7
  ret float %op
}

; CHECK-LABEL: @maximum_nsz(
; RESULT-NEXT: %op = fmul nsz float %a, %b, !dbg !7
; RESULT-NEXT: ret
define float @maximum_nsz(float %a, float %b) {
  %op = call nsz float @llvm.maximum.f32(float %a, float %b), !dbg !7
  ret float %op
}

; CHECK-LABEL: @sqrt_ninf(
; RESULT-NEXT: %op = fmul ninf float %a, 2.000000e+00, !dbg !7
; RESULT-NEXT: ret
define float @sqrt_ninf(float %a, float %b) {
  %op = call ninf float @llvm.sqrt.f32(float %a), !dbg !7
  ret float %op
}

; CHECK-LABEL: @sqrt_vec(
; RESULT-NEXT: %op = fmul <2 x float> %a, <float 2.000000e+00, float 2.000000e+00>, !dbg !7
; RESULT-NEXT: ret
define <2 x float> @sqrt_vec(<2 x float> %a, <2 x float> %b) {
  %op = call <2 x float> @llvm.sqrt.v2f32(<2 x float> %a), !dbg !7
  ret <2 x float> %op
}

; CHECK-LABEL: @div_fixup(
; RESULT-NEXT: %op = call float @llvm.fma.f32(float %a, float %b, float %c)
; RESULT-NEXT: ret
define float @div_fixup(float %a, float %b, float %c) {
  %op = call float @llvm.amdgcn.div.fixup.f32(float %a, float %b, float %c)
  ret float %op
}

; CHECK-LABEL: @fma_legacy(
; RESULT-NEXT: %op = call float @llvm.fma.f32(float %a, float %b, float %c)
; RESULT-NEXT: ret
define float @fma_legacy(float %a, float %b, float %c) {
  %op = call float @llvm.amdgcn.fma.legacy(float %a, float %b, float %c)
  ret float %op
}

; CHECK-LABEL: @fmul_legacy(
; RESULT-NEXT: %op = fmul float %a, %b
; RESULT-NEXT: ret
define float @fmul_legacy(float %a, float %b) {
  %op = call float @llvm.amdgcn.fmul.legacy(float %a, float %b)
  ret float %op
}

declare i32 @llvm.amdgcn.workitem.id.y()
declare i32 @llvm.amdgcn.workitem.id.z()
declare i32 @llvm.amdgcn.workgroup.id.y()
declare i32 @llvm.amdgcn.workgroup.id.z()
declare float @llvm.amdgcn.div.fixup.f32(float, float, float)
declare float @llvm.amdgcn.fma.legacy(float, float, float)
declare float @llvm.amdgcn.fmul.legacy(float, float)

declare float @llvm.sqrt.f32(float)
declare <2 x float> @llvm.sqrt.v2f32(<2 x float>)
declare float @llvm.maxnum.f32(float, float)
declare float @llvm.minnum.f32(float, float)
declare float @llvm.maximum.f32(float, float)
declare float @llvm.minimum.f32(float, float)

!llvm.dbg.cu = !{!0}
!opencl.ocl.version = !{!3, !3}
!llvm.module.flags = !{!4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "arst.c", directory: "/some/random/directory")
!2 = !{}
!3 = !{i32 2, i32 0}
!4 = !{i32 2, !"Dwarf Version", i32 2}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{!""}
!7 = !DILocation(line: 2, column: 6, scope: !8)
!8 = distinct !DISubprogram(name: "arst", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{float 2.500000e+00}
