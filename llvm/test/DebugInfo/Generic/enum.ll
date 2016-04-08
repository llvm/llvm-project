; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s

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

@a = global i64 0, align 8

; Function Attrs: nounwind uwtable
define void @_Z4funcv() #0 !dbg !13 {
entry:
  %b = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %b, metadata !20, metadata !DIExpression()), !dbg !22
  store i32 0, i32* %b, align 4, !dbg !22
  ret void, !dbg !23
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!19, !24}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.4 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !11, subprograms: !12, globals: !17, imports: !11)
!1 = !DIFile(filename: "enum.cpp", directory: "/tmp")
!2 = !{!3, !8}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "e1", line: 1, size: 64, align: 64, file: !1, elements: !4)
!4 = !{!5, !6, !7}
!5 = !DIEnumerator(name: "I", value: 0) ; [ DW_TAG_enumerator ] [I :: 0]
!6 = !DIEnumerator(name: "J", value: 4294967295) ; [ DW_TAG_enumerator ] [J :: 4294967295]
!7 = !DIEnumerator(name: "K", value: -1152921504606846976) ; [ DW_TAG_enumerator ] [K :: 17293822569102704640]
!8 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "e2", line: 2, size: 32, align: 32, file: !1, elements: !9)
!9 = !{!10}
!10 = !DIEnumerator(name: "X", value: 0) ; [ DW_TAG_enumerator ] [X :: 0]
!11 = !{}
!12 = !{!13}
!13 = distinct !DISubprogram(name: "func", linkageName: "_Z4funcv", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 3, file: !1, scope: !14, type: !15, variables: !11)
!14 = !DIFile(filename: "enum.cpp", directory: "/tmp")
!15 = !DISubroutineType(types: !16)
!16 = !{null}
!17 = !{!18}
!18 = !DIGlobalVariable(name: "a", line: 1, isLocal: false, isDefinition: true, scope: null, file: !14, type: !3, variable: i64* @a)
!19 = !{i32 2, !"Dwarf Version", i32 3}
!20 = !DILocalVariable(name: "b", line: 4, scope: !13, file: !14, type: !21)
!21 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!22 = !DILocation(line: 4, scope: !13)
!23 = !DILocation(line: 5, scope: !13)
!24 = !{i32 1, !"Debug Info Version", i32 3}
