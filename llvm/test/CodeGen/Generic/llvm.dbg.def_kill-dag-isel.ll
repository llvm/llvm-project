; RUN: llc -O0 -march=amdgcn -stop-after=finalize-isel < %s 2>&1 | FileCheck %s

; CHECK: warning: ignoring debug info with an invalid version (4) in <stdin>

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

define void @undef() {
entry:
  call void @llvm.dbg.def(metadata !3, metadata i1 undef), !dbg !5
  call void @llvm.dbg.kill(metadata !3), !dbg !7
  unreachable
}

define void @i1() {
entry:
  call void @llvm.dbg.def(metadata !8, metadata i1 true), !dbg !5
  call void @llvm.dbg.kill(metadata !8), !dbg !7
  unreachable
}

define void @i128() {
entry:
  call void @llvm.dbg.def(metadata !9, metadata i128 36893488147419103232), !dbg !5
  call void @llvm.dbg.kill(metadata !9), !dbg !7
  unreachable
}

define void @float() {
entry:
  call void @llvm.dbg.def(metadata !10, metadata float 1.000000e+00), !dbg !5
  call void @llvm.dbg.kill(metadata !10), !dbg !7
  unreachable
}

define void @alloca() {
entry:
  %0 = alloca i1, addrspace(5)
  call void @llvm.dbg.def(metadata !11, metadata ptr addrspace(5) %0), !dbg !5
  call void @llvm.dbg.kill(metadata !11), !dbg !7
  unreachable
}

define void @unused_argument(i32 %0) {
entry:
  call void @llvm.dbg.def(metadata !12, metadata i32 %0), !dbg !5
  call void @llvm.dbg.kill(metadata !12), !dbg !7
  unreachable
}

define i32 @used_argument(i32 %0) {
entry:
  call void @llvm.dbg.def(metadata !13, metadata i32 %0), !dbg !5
  call void @llvm.dbg.kill(metadata !13), !dbg !7
  ret i32 %0
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.def(metadata, metadata) #0

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.kill(metadata) #0

attributes #0 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!2}
!llvm.dbg.cu = !{!0}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug)
!1 = !DIFile(filename: "<stdin>", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 4}
!3 = distinct !DILifetime(object: !4, location: !DIExpr(DIOpReferrer(i1)))
!4 = distinct !DIFragment()
!5 = !DILocation(line: 4, column: 1, scope: !6)
!6 = distinct !DISubprogram(scope: null, spFlags: DISPFlagDefinition, unit: !0)
!7 = !DILocation(line: 8, column: 1, scope: !6)
!8 = distinct !DILifetime(object: !4, location: !DIExpr(DIOpReferrer(i1)))
!9 = distinct !DILifetime(object: !4, location: !DIExpr(DIOpReferrer(i128)))
!10 = distinct !DILifetime(object: !4, location: !DIExpr(DIOpReferrer(float)))
!11 = distinct !DILifetime(object: !4, location: !DIExpr(DIOpReferrer(i1 addrspace(5)*), DIOpDeref(i1)))
!12 = distinct !DILifetime(object: !4, location: !DIExpr(DIOpReferrer(i32)))
!13 = distinct !DILifetime(object: !4, location: !DIExpr(DIOpReferrer(i32)))
