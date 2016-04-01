; RUN: llc -O0 -march=amdgcn -mtriple=amdgcn-unknown-amdhsa -verify-machineinstrs -mattr=-flat-for-global < %s | FileCheck %s

; CHECK-LABEL: {{^}}test_debug_value:
; CHECK: s_load_dwordx2 s[4:5]
; CHECK: DEBUG_VALUE: test_debug_value:globalptr_arg <- %SGPR4_SGPR5
; CHECK: buffer_store_dword
; CHECK: s_endpgm
define void @test_debug_value(i32 addrspace(1)* nocapture %globalptr_arg) #0 !dbg !4 {
entry:
  tail call void @llvm.dbg.value(metadata i32 addrspace(1)* %globalptr_arg, i64 0, metadata !10, metadata !13), !dbg !14
  store i32 123, i32 addrspace(1)* %globalptr_arg, align 4
  ret void
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0 (trunk 244715) (llvm/trunk 244718)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, subprograms: !3)
!1 = !DIFile(filename: "/tmp/test_debug_value.cl", directory: "/Users/matt/src/llvm/build_debug")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "test_debug_value", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, variables: !9)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7}
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64, align: 32)
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{!10}
!10 = !DILocalVariable(name: "globalptr_arg", arg: 1, scope: !4, file: !1, line: 1, type: !7)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !DIExpression()
!14 = !DILocation(line: 1, column: 42, scope: !4)
