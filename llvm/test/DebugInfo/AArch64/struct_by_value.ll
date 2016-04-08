; A by-value struct is a register-indirect value (breg).
; RUN: llc %s -filetype=asm -o - | FileCheck %s

; CHECK: Lsection_info:
; CHECK: DW_AT_location
; CHECK-NEXT: .byte 112
; 112 = 0x70 = DW_OP_breg0

; rdar://problem/13658587
;
; Generated from
;
; struct five
; {
;   int a;
;   int b;
;   int c;
;   int d;
;   int e;
; };
;
; int
; return_five_int (struct five f)
; {
;   return f.a;
; }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n32:64-S128"
target triple = "arm64-apple-ios3.0.0"

%struct.five = type { i32, i32, i32, i32, i32 }

; Function Attrs: nounwind ssp
define i32 @return_five_int(%struct.five* %f) #0 !dbg !4 {
entry:
  call void @llvm.dbg.declare(metadata %struct.five* %f, metadata !17, metadata !DIExpression(DW_OP_deref)), !dbg !18
  %a = getelementptr inbounds %struct.five, %struct.five* %f, i32 0, i32 0, !dbg !19
  %0 = load i32, i32* %a, align 4, !dbg !19
  ret i32 %0, !dbg !19
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind ssp }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!16, !20}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "LLVM version 3.4 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "struct_by_value.c", directory: "")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "return_five_int", line: 13, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 14, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "struct_by_value.c", directory: "")
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DICompositeType(tag: DW_TAG_structure_type, name: "five", line: 1, size: 160, align: 32, file: !1, elements: !10)
!10 = !{!11, !12, !13, !14, !15}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "a", line: 3, size: 32, align: 32, file: !1, scope: !9, baseType: !8)
!12 = !DIDerivedType(tag: DW_TAG_member, name: "b", line: 4, size: 32, align: 32, offset: 32, file: !1, scope: !9, baseType: !8)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "c", line: 5, size: 32, align: 32, offset: 64, file: !1, scope: !9, baseType: !8)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "d", line: 6, size: 32, align: 32, offset: 96, file: !1, scope: !9, baseType: !8)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "e", line: 7, size: 32, align: 32, offset: 128, file: !1, scope: !9, baseType: !8)
!16 = !{i32 2, !"Dwarf Version", i32 2}
!17 = !DILocalVariable(name: "f", line: 13, arg: 1, scope: !4, file: !5, type: !9)
!18 = !DILocation(line: 13, scope: !4)
!19 = !DILocation(line: 16, scope: !4)
!20 = !{i32 1, !"Debug Info Version", i32 3}
