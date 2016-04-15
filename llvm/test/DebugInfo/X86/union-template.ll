; RUN: llc -O0 -mtriple=x86_64-linux-gnu %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; Verify that we've emitted template arguments for the union
; CHECK: DW_TAG_union_type
; CHECK-NEXT: "Value<float>"
; CHECK: DW_TAG_template_type_parameter
; CHECK: "T"

%"union.PR15637::Value" = type { i32 }

@_ZN7PR156371fE = global %"union.PR15637::Value" zeroinitializer, align 4

define void @_ZN7PR156371gEf(float %value) #0 !dbg !4 {
entry:
  %value.addr = alloca float, align 4
  %tempValue = alloca %"union.PR15637::Value", align 4
  store float %value, float* %value.addr, align 4
  call void @llvm.dbg.declare(metadata float* %value.addr, metadata !23, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.declare(metadata %"union.PR15637::Value"* %tempValue, metadata !25, metadata !DIExpression()), !dbg !26
  ret void, !dbg !27
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!28}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.3 (trunk 178499) (llvm/trunk 178472)", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !9, imports:  !2)
!1 = !DIFile(filename: "foo.cc", directory: "/usr/local/google/home/echristo/tmp")
!2 = !{}
!4 = distinct !DISubprogram(name: "g", linkageName: "_ZN7PR156371gEf", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 3, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DINamespace(name: "PR15637", line: 1, file: !1, scope: null)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!9 = !{!10}
!10 = !DIGlobalVariable(name: "f", linkageName: "_ZN7PR156371fE", line: 6, isLocal: false, isDefinition: true, scope: !5, file: !11, type: !12, variable: %"union.PR15637::Value"* @_ZN7PR156371fE)
!11 = !DIFile(filename: "foo.cc", directory: "/usr/local/google/home/echristo/tmp")
!12 = !DICompositeType(tag: DW_TAG_union_type, name: "Value<float>", line: 2, size: 32, align: 32, file: !1, scope: !5, elements: !13, templateParams: !21)
!13 = !{!14, !16}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "a", line: 2, size: 32, align: 32, file: !1, scope: !12, baseType: !15)
!15 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!16 = !DISubprogram(name: "Value", line: 2, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, scopeLine: 2, file: !1, scope: !12, type: !17)
!17 = !DISubroutineType(types: !18)
!18 = !{null, !19}
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !12)
!21 = !{!22}
!22 = !DITemplateTypeParameter(name: "T", type: !8)
!23 = !DILocalVariable(name: "value", line: 3, arg: 1, scope: !4, file: !11, type: !8)
!24 = !DILocation(line: 3, scope: !4)
!25 = !DILocalVariable(name: "tempValue", line: 4, scope: !4, file: !11, type: !12)
!26 = !DILocation(line: 4, scope: !4)
!27 = !DILocation(line: 5, scope: !4)
!28 = !{i32 1, !"Debug Info Version", i32 3}
