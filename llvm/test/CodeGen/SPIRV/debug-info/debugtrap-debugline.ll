; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - -filetype=obj | spirv-val %}

; A debugtrap with a location must get its DebugLine before the DebugBreak.

; CHECK-DAG:  [[DI:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG:  [[DB:%[0-9]+]] = OpExtInstImport "NonSemantic.DebugBreak"
; CHECK-DAG:  [[C2:%[0-9]+]] = OpConstant {{%[0-9]+}} 2{{$}}
; CHECK:      OpExtInst [[VOID:%[0-9]+]] [[DI]] DebugLine {{%[0-9]+}} [[C2]] [[C2]]
; CHECK-NEXT: OpExtInst [[VOID]] [[DB]] DebugBreak

define spir_func void @f() !dbg !5 {
entry:
  call void @llvm.debugtrap(), !dbg !8
  ret void, !dbg !9
}

declare void @llvm.debugtrap()

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_OpenCL, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cl", directory: "/src")
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !DILocation(line: 2, column: 3, scope: !5)
!9 = !DILocation(line: 3, column: 1, scope: !5)
