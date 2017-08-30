; RUN: llc -mtriple=x86_64-apple-darwin < %s -filetype=obj \
; RUN:     | llvm-dwarfdump -debug-dump=info - | FileCheck --check-prefix=CHECK --check-prefix=DARWIN %s
; RUN: llc -mtriple=x86_64-linux-gnu < %s -filetype=obj \
; RUN:     | llvm-dwarfdump -debug-dump=info - | FileCheck --check-prefix=CHECK --check-prefix=LINUX %s
; RUN: llc -mtriple=x86_64-apple-darwin < %s -filetype=obj -regalloc=basic \
; RUN:     | llvm-dwarfdump -debug-dump=info - | FileCheck --check-prefix=CHECK --check-prefix=DARWIN %s

; CHECK: DW_TAG_subprogram
; CHECK:   DW_AT_abstract_origin {{.*}} "foo"
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_abstract_origin {{.*}} "sp"
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_abstract_origin {{.*}} "nums"

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "foo"
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "sp"
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "nums"

;CHECK: DW_TAG_inlined_subroutine
;CHECK-NEXT: DW_AT_abstract_origin {{.*}} "foo"
;CHECK-NEXT: DW_AT_low_pc [DW_FORM_addr]
;CHECK-NEXT: DW_AT_high_pc [DW_FORM_data4]
;CHECK-NEXT: DW_AT_call_file
;CHECK-NEXT: DW_AT_call_line

;CHECK: DW_TAG_formal_parameter
;FIXME: Linux shouldn't drop this parameter either...
;CHECK-NOT: DW_TAG
;DARWIN:   DW_AT_abstract_origin {{.*}} "sp"
;DARWIN: DW_TAG_formal_parameter
;CHECK: DW_AT_abstract_origin {{.*}} "nums"
;CHECK-NOT: DW_TAG_formal_parameter

source_filename = "test/DebugInfo/X86/dbg-value-inlined-parameter.ll"

%struct.S1 = type { float*, i32 }

@p = common global %struct.S1 zeroinitializer, align 8, !dbg !0

; Function Attrs: nounwind optsize ssp
define i32 @foo(%struct.S1* nocapture %sp, i32 %nums) #0 !dbg !15 {
entry:
  tail call void @llvm.dbg.value(metadata %struct.S1* %sp, metadata !19, metadata !22), !dbg !23
  tail call void @llvm.dbg.value(metadata i32 %nums, metadata !21, metadata !22), !dbg !24
  %tmp2 = getelementptr inbounds %struct.S1, %struct.S1* %sp, i64 0, i32 1, !dbg !25
  store i32 %nums, i32* %tmp2, align 4, !dbg !25
  %call = tail call float* @bar(i32 %nums) #3, !dbg !27
  %tmp5 = getelementptr inbounds %struct.S1, %struct.S1* %sp, i64 0, i32 0, !dbg !27
  store float* %call, float** %tmp5, align 8, !dbg !27
  %cmp = icmp ne float* %call, null, !dbg !28
  %cond = zext i1 %cmp to i32, !dbg !28
  ret i32 %cond, !dbg !28
}

; Function Attrs: optsize
declare float* @bar(i32) #1

; Function Attrs: nounwind optsize ssp
define void @foobar() #0 !dbg !29 {
entry:
  tail call void @llvm.dbg.value(metadata %struct.S1* @p, metadata !19, metadata !22) #4, !dbg !32
  tail call void @llvm.dbg.value(metadata i32 1, metadata !21, metadata !22) #4, !dbg !35
  store i32 1, i32* getelementptr inbounds (%struct.S1, %struct.S1* @p, i64 0, i32 1), align 8, !dbg !36
  %call.i = tail call float* @bar(i32 1) #3, !dbg !37
  store float* %call.i, float** getelementptr inbounds (%struct.S1, %struct.S1* @p, i64 0, i32 0), align 8, !dbg !37
  ret void, !dbg !38
}

; Function Attrs: nounwind readnone

declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind optsize ssp }
attributes #1 = { optsize }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind optsize }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!14}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "p", scope: !2, file: !3, line: 14, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 2.9 (trunk 125693)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !5, imports: !4)
!3 = !DIFile(filename: "nm2.c", directory: "/private/tmp")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_typedef, name: "S1", scope: !2, file: !3, line: 4, baseType: !7)
!7 = !DICompositeType(tag: DW_TAG_structure_type, name: "S1", scope: !2, file: !3, line: 1, size: 128, align: 64, elements: !8)
!8 = !{!9, !12}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "m", scope: !3, file: !3, line: 2, baseType: !10, size: 64, align: 64)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, scope: !2, baseType: !11, size: 64, align: 64)
!11 = !DIBasicType(name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!12 = !DIDerivedType(tag: DW_TAG_member, name: "nums", scope: !3, file: !3, line: 3, baseType: !13, size: 32, align: 32, offset: 64)
!13 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!14 = !{i32 1, !"Debug Info Version", i32 3}
!15 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 8, type: !16, isLocal: false, isDefinition: true, scopeLine: 8, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !2, variables: !18)
!16 = !DISubroutineType(types: !17)
!17 = !{!13}
!18 = !{!19, !21}
!19 = !DILocalVariable(name: "sp", arg: 1, scope: !15, file: !3, line: 7, type: !20)
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, scope: !2, baseType: !6, size: 64, align: 64)
!21 = !DILocalVariable(name: "nums", arg: 2, scope: !15, file: !3, line: 7, type: !13)
!22 = !DIExpression()
!23 = !DILocation(line: 7, column: 13, scope: !15)
!24 = !DILocation(line: 7, column: 21, scope: !15)
!25 = !DILocation(line: 9, column: 3, scope: !26)
!26 = distinct !DILexicalBlock(scope: !15, file: !3, line: 8, column: 1)
!27 = !DILocation(line: 10, column: 3, scope: !26)
!28 = !DILocation(line: 11, column: 3, scope: !26)
!29 = distinct !DISubprogram(name: "foobar", scope: !3, file: !3, line: 15, type: !30, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !2)
!30 = !DISubroutineType(types: !31)
!31 = !{null}
!32 = !DILocation(line: 7, column: 13, scope: !15, inlinedAt: !33)
!33 = !DILocation(line: 16, column: 3, scope: !34)
!34 = distinct !DILexicalBlock(scope: !29, file: !3, line: 15, column: 15)
!35 = !DILocation(line: 7, column: 21, scope: !15, inlinedAt: !33)
!36 = !DILocation(line: 9, column: 3, scope: !26, inlinedAt: !33)
!37 = !DILocation(line: 10, column: 3, scope: !26, inlinedAt: !33)
!38 = !DILocation(line: 17, column: 1, scope: !34)

