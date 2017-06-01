; RUN: llc -o - %s | FileCheck %s
; This test checks that parameters on the stack pointer are correctly
; referenced by debug info.
target triple = "x86_64--"

@glob = external global i64
@ptr = external global i32*
%struct.s = type { i32, i32, i32, i32, i32 }

; CHECK-LABEL: f0:
; CHECK: DEBUG_VALUE: f:input <- [%RSP+8]
define i32 @f0(%struct.s* byval align 8 %input) !dbg !8 {
  call void @llvm.dbg.declare(metadata %struct.s* %input, metadata !4, metadata !17), !dbg !18
  ret i32 42
}

; CHECK-LABEL: f1:
; CHECK: DEBUG_VALUE: f:input <- [%RBP+16]
define i32 @f1(%struct.s* byval align 8 %input) !dbg !19 {
  %val = load i64, i64* @glob
  ; this alloca should force FP usage.
  %stackspace = alloca i32, i64 %val, align 1
  store i32* %stackspace, i32** @ptr
  call void @llvm.dbg.declare(metadata %struct.s* %input, metadata !20, metadata !17), !dbg !21
  ret i32 42
}

; CHECK-LABEL: f2:
; Just check that we are indeed aligning the stack and setting up a base pointer
; in RBX.
; CHECK: pushq %rbp
; CHECK: movq %rsp, %rbp
; CHECK: pushq %rbx
; CHECK: andq $-64, %rsp
; CHECK: subq $64, %rsp
; CHECK: movq %rsp, %rbx
; The parameter should still be referenced through RBP though.
; CHECK-NOT: DEBUG_VALUE: f:input <- [%RBX
; CHECK: DEBUG_VALUE: f:input <- [%RBP+16]
define i32 @f2(%struct.s* byval align 8 %input) !dbg !22 {
  %val = load i64, i64* @glob
  %stackspace = alloca i32, i64 %val, align 64
  store i32* %stackspace, i32** @ptr
  call void @llvm.dbg.declare(metadata %struct.s* %input, metadata !23, metadata !17), !dbg !24
  ret i32 42
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3)
!3 = !DIFile(filename: "dbg-baseptr.ll", directory: "/")
!4 = !DILocalVariable(name: "input", arg: 1, scope: !8, file: !3, line: 5, type: !9)
!5 = !{}

!6 = !DISubroutineType(types: !7)
!7 = !{!10, !9}

!8 = distinct !DISubprogram(name: "f", file: !3, line: 5, type: !6, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, unit: !2, variables: !5)

!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s", elements: !11)
!10 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!11 = !{!12, !13, !14, !15, !16}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "a", baseType: !10, size: 32)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "b", baseType: !10, size: 32, offset: 32)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "c", baseType: !10, size: 32, offset: 64)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "d", baseType: !10, size: 32, offset: 96)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "e", baseType: !10, size: 32, offset: 128)

!17 = !DIExpression()
!18 = !DILocation(line: 5, scope: !8)

!19 = distinct !DISubprogram(name: "f", file: !3, line: 5, type: !6, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, unit: !2, variables: !5)
!20 = !DILocalVariable(name: "input", arg: 1, scope: !19, file: !3, line: 5, type: !9)
!21 = !DILocation(line: 5, scope: !19)
!22 = distinct !DISubprogram(name: "f", file: !3, line: 5, type: !6, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, unit: !2, variables: !5)
!23 = !DILocalVariable(name: "input", arg: 1, scope: !22, file: !3, line: 5, type: !9)
!24 = !DILocation(line: 5, scope: !22)
