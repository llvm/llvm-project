; RUN: llc -mtriple=x86_64-macosx %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -v -debug-info %t | FileCheck %s

; CHECK: DW_TAG_subprogram [9] *
; CHECK-NOT: DW_AT_{{(MIPS_)?}}linkage_name
; CHECK: DW_AT_specification

source_filename = "test/DebugInfo/X86/linkage-name.ll"

%class.A = type { i8 }

@a = global %class.A zeroinitializer, align 1, !dbg !0

; Function Attrs: nounwind ssp uwtable
define i32 @_ZN1A1aEi(ptr %this, i32 %b) #0 align 2 !dbg !14 {
entry:
  %this.addr = alloca ptr, align 8
  %b.addr = alloca i32, align 4
  store ptr %this, ptr %this.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %this.addr, metadata !15, metadata !17), !dbg !18
  store i32 %b, ptr %b.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %b.addr, metadata !19, metadata !17), !dbg !20
  %this1 = load ptr, ptr %this.addr
  %0 = load i32, ptr %b.addr, align 4, !dbg !21
  ret i32 %0, !dbg !21
}

; Function Attrs: nounwind readnone

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!10}
!llvm.module.flags = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "a", scope: null, file: !2, line: 9, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "foo.cpp", directory: "/Users/echristo")
!3 = !DICompositeType(tag: DW_TAG_class_type, name: "A", file: !2, line: 1, size: 8, align: 8, elements: !4)
!4 = !{!5}
!5 = !DISubprogram(name: "a", linkageName: "_ZN1A1aEi", scope: !3, file: !2, line: 2, type: !6, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrivate | DIFlagPrototyped, isOptimized: false)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9, !8}
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64, align: 64, flags: DIFlagArtificial)
!10 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 3.1 (trunk 152691) (llvm/trunk 152692)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !11, retainedTypes: !11, globals: !12, imports: !11)
!11 = !{}
!12 = !{!0}
!13 = !{i32 1, !"Debug Info Version", i32 3}
!14 = distinct !DISubprogram(name: "a", linkageName: "_ZN1A1aEi", scope: null, file: !2, line: 5, type: !6, isLocal: false, isDefinition: true, scopeLine: 5, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !10, declaration: !5)
!15 = !DILocalVariable(name: "this", arg: 1, scope: !14, file: !2, line: 5, type: !16, flags: DIFlagArtificial)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64, align: 64)
!17 = !DIExpression()
!18 = !DILocation(line: 5, column: 8, scope: !14)
!19 = !DILocalVariable(name: "b", arg: 2, scope: !14, file: !2, line: 5, type: !8)
!20 = !DILocation(line: 5, column: 14, scope: !14)
!21 = !DILocation(line: 6, column: 4, scope: !22)
!22 = distinct !DILexicalBlock(scope: !14, file: !2, line: 5, column: 17)

