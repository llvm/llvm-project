; When a call to a deprecated buffer intrinsic carries a debug location, the
; deprecation warning reports it, so a user of a large module has somewhere to
; go. The diagnostic is still emitted once per intrinsic, not once per call.

; RUN: llc -mtriple=amdgpu9.00-amd-mesa3d -filetype=null < %s 2>&1 | \
; RUN:   FileCheck --implicit-check-not=warning: %s

; Reported against the first call that has a location, not the one without, and
; not twice even though there are three calls.
; CHECK: warning: shader.cl:42:7: 'llvm.amdgcn.raw.buffer.load' is deprecated {{.*}} use 'llvm.amdgcn.raw.ptr.buffer.load' {{.*}}

; An intrinsic whose calls carry no location still warns, without picking up a
; useless "<unknown>:0:0" prefix.
; CHECK: warning: 'llvm.amdgcn.s.buffer.load' is deprecated {{.*}} use 'llvm.amdgcn.ptr.s.buffer.load' {{.*}}

define amdgpu_ps float @no_loc_first(<4 x i32> inreg %rsrc, i32 %off) !dbg !4 {
  %a = call float @llvm.amdgcn.raw.buffer.load.f32(<4 x i32> %rsrc, i32 %off, i32 0, i32 0)
  %b = call float @llvm.amdgcn.raw.buffer.load.f32(<4 x i32> %rsrc, i32 %off, i32 4, i32 0), !dbg !7
  %c = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 0, i32 0)
  %cf = bitcast i32 %c to float
  %s0 = fadd float %a, %b
  %s1 = fadd float %s0, %cf
  ret float %s1
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "shader.cl", directory: "/src")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "no_loc_first", scope: !1, file: !1, line: 10, type: !5, scopeLine: 10, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 42, column: 7, scope: !4)
