; REQUIRES: asserts
; RUN: opt -S -passes=amdgpu-simplifylib -debug-only=amdgpu-simplifylib -mtriple=amdgcn-unknown-amdhsa -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt -S -passes=amdgpu-simplifylib -debug-only=amdgpu-simplifylib -mtriple=amdgcn-unknown-amdhsa -disable-output < %s 2>&1 | FileCheck %s

; CHECK-NOT: AMDIC: try folding   call void @llvm.lifetime.start.p0
; CHECK-NOT: AMDIC: try folding   call void @llvm.lifetime.end.p0
; CHECK-NOT: AMDIC: try folding   call void @llvm.dbg.value

define void @foo(i32 %i) {
  call void @llvm.lifetime.start.p0(i64 1, ptr undef)
  call void @llvm.lifetime.end.p0(i64 1, ptr undef)
  call void @llvm.dbg.value(metadata i32 undef, metadata !DILocalVariable(name: "1", scope: !2), metadata !DIExpression()), !dbg !3
  ret void
}

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.module.flags = !{!1}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !DIFile(filename: "1", directory: "1"))
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DISubprogram(unit: !0)
!3 = !DILocation(line: 1, column: 1, scope: !2)
