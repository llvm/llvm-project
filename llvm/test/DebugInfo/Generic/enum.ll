; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump -v %t | FileCheck %s

; IR generated from the following code compiled with clang -g:
; enum e1 { I, J = 0xffffffffU, K = 0xf000000000000000ULL } a;
; enum e2 { X };
; void func() {
;   int b = X;
; }

; These values were previously being truncated to -1 and 0 respectively.

; CHECK: debug_info contents
; CHECK: DW_TAG_enumeration_type
; CHECK-NEXT: DW_AT_name{{.*}} = "e1"
; CHECK-NOT: NULL
; CHECK: DW_TAG_enumerator
; CHECK-NOT: NULL
; CHECK: DW_TAG_enumerator
; CHECK-NEXT: DW_AT_name{{.*}} = "J"
; CHECK-NEXT: DW_AT_const_value [DW_FORM_sdata]     (4294967295)
; CHECK-NOT: NULL
; CHECK: DW_TAG_enumerator
; CHECK-NEXT: DW_AT_name{{.*}} = "K"
; CHECK-NEXT: DW_AT_const_value [DW_FORM_sdata]     (-1152921504606846976)

; Check that we retain enums that aren't referenced by any variables, etc
; CHECK: DW_TAG_enumeration_type
; CHECK-NEXT: DW_AT_name{{.*}} = "e2"
; CHECK-NOT: NULL
; CHECK: DW_TAG_enumerator
; CHECK-NEXT: DW_AT_name{{.*}} = "X"

source_filename = "test/DebugInfo/Generic/enum.ll"

@a = global i64 0, align 8, !dbg !0

; Function Attrs: nounwind uwtable
define void @_Z4funcv() #0 !dbg !17 {
entry:
  %b = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %b, metadata !20, metadata !22), !dbg !23
  store i32 0, i32* %b, align 4, !dbg !23
  ret void, !dbg !24
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!8}
!llvm.module.flags = !{!15, !16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "a", scope: null, file: !2, line: 1, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "enum.cpp", directory: "/tmp")
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "e1", file: !2, line: 1, size: 64, align: 64, elements: !4)
!4 = !{!5, !6, !7}
!5 = !DIEnumerator(name: "I", value: 0)
!6 = !DIEnumerator(name: "J", value: 4294967295) ; [ DW_TAG_enumerator ] [I :: 0]
!7 = !DIEnumerator(name: "K", value: -1152921504606846976) ; [ DW_TAG_enumerator ] [J :: 4294967295]
!8 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 3.4 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !9, retainedTypes: !13, globals: !14, imports: !13) ; [ DW_TAG_enumerator ] [K :: 17293822569102704640]
!9 = !{!3, !10}
!10 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "e2", file: !2, line: 2, size: 32, align: 32, elements: !11)
!11 = !{!12}
!12 = !DIEnumerator(name: "X", value: 0) ; [ DW_TAG_enumerator ] [X :: 0]
!13 = !{}
!14 = !{!0}
!15 = !{i32 2, !"Dwarf Version", i32 3}
!16 = !{i32 1, !"Debug Info Version", i32 3}
!17 = distinct !DISubprogram(name: "func", linkageName: "_Z4funcv", scope: !2, file: !2, line: 3, type: !18, isLocal: false, isDefinition: true, scopeLine: 3, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !8, variables: !13)
!18 = !DISubroutineType(types: !19)
!19 = !{null}
!20 = !DILocalVariable(name: "b", scope: !17, file: !2, line: 4, type: !21)
!21 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!22 = !DIExpression()
!23 = !DILocation(line: 4, scope: !17)
!24 = !DILocation(line: 5, scope: !17)

