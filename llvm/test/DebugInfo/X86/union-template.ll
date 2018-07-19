; RUN: llc -O0 -mtriple=x86_64-linux-gnu %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s

; Verify that we've emitted template arguments for the union
; CHECK: DW_TAG_union_type
; CHECK-NEXT: "Value<float>"
; CHECK: DW_TAG_template_type_parameter
; CHECK: "T"

source_filename = "test/DebugInfo/X86/union-template.ll"

%"union.PR15637::Value" = type { i32 }

@_ZN7PR156371fE = global %"union.PR15637::Value" zeroinitializer, align 4, !dbg !0

; Function Attrs: nounwind
define void @_ZN7PR156371gEf(float %value) #0 !dbg !19 {
entry:
  %value.addr = alloca float, align 4
  %tempValue = alloca %"union.PR15637::Value", align 4
  store float %value, float* %value.addr, align 4
  call void @llvm.dbg.declare(metadata float* %value.addr, metadata !22, metadata !23), !dbg !24
  call void @llvm.dbg.declare(metadata %"union.PR15637::Value"* %tempValue, metadata !25, metadata !23), !dbg !26
  ret void, !dbg !27
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!15}
!llvm.module.flags = !{!18}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "f", linkageName: "_ZN7PR156371fE", scope: !2, file: !3, line: 6, type: !4, isLocal: false, isDefinition: true)
!2 = !DINamespace(name: "PR15637", scope: null)
!3 = !DIFile(filename: "foo.cc", directory: "/usr/local/google/home/echristo/tmp")
!4 = !DICompositeType(tag: DW_TAG_union_type, name: "Value<float>", scope: !2, file: !3, line: 2, size: 32, align: 32, elements: !5, templateParams: !12)
!5 = !{!6, !8}
!6 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !4, file: !3, line: 2, baseType: !7, size: 32, align: 32)
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !DISubprogram(name: "Value", scope: !4, file: !3, line: 2, type: !9, isLocal: false, isDefinition: false, scopeLine: 2, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!12 = !{!13}
!13 = !DITemplateTypeParameter(name: "T", type: !14)
!14 = !DIBasicType(name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!15 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 3.3 (trunk 178499) (llvm/trunk 178472)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !16, retainedTypes: !16, globals: !17, imports: !16)
!16 = !{}
!17 = !{!0}
!18 = !{i32 1, !"Debug Info Version", i32 3}
!19 = distinct !DISubprogram(name: "g", linkageName: "_ZN7PR156371gEf", scope: !2, file: !3, line: 3, type: !20, isLocal: false, isDefinition: true, scopeLine: 3, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !15, variables: !16)
!20 = !DISubroutineType(types: !21)
!21 = !{null, !14}
!22 = !DILocalVariable(name: "value", arg: 1, scope: !19, file: !3, line: 3, type: !14)
!23 = !DIExpression()
!24 = !DILocation(line: 3, scope: !19)
!25 = !DILocalVariable(name: "tempValue", scope: !19, file: !3, line: 4, type: !4)
!26 = !DILocation(line: 4, scope: !19)
!27 = !DILocation(line: 5, scope: !19)

