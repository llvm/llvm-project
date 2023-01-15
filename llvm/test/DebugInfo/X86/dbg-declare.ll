; RUN: llc < %s -O0 -mtriple x86_64-apple-darwin | FileCheck %s
; RUN: llc < %s -O0 -mtriple x86_64-apple-darwin -filetype=obj \
; RUN:     | llvm-dwarfdump -v - --debug-info | FileCheck %s --check-prefix=DWARF
; <rdar://problem/11134152>

; CHECK-LABEL: _foo:
; CHECK-NOT: #DEBUG_VALUE

; "[DW_FORM_exprloc] <0x2> 91 XX" means fbreg uleb(XX)
; DWARF-LABEL: DW_TAG_formal_parameter
; DWARF-NEXT:              DW_AT_location [DW_FORM_exprloc]      (DW_OP_fbreg -16)
; DWARF-NEXT:              DW_AT_name [DW_FORM_strp]     ( {{.*}} = "x")

; FIXME: There is no debug info to describe "a".

define i32 @foo(ptr %x) nounwind uwtable ssp !dbg !5 {
entry:
  %x.addr = alloca ptr, align 8
  %saved_stack = alloca ptr
  %cleanup.dest.slot = alloca i32
  store ptr %x, ptr %x.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %x.addr, metadata !14, metadata !DIExpression()), !dbg !15
  %0 = load ptr, ptr %x.addr, align 8, !dbg !16
  %1 = load i32, ptr %0, align 4, !dbg !16
  %2 = zext i32 %1 to i64, !dbg !16
  %3 = call ptr @llvm.stacksave(), !dbg !16
  store ptr %3, ptr %saved_stack, !dbg !16
  %vla = alloca i8, i64 %2, align 16, !dbg !16
  call void @llvm.dbg.declare(metadata ptr %vla, metadata !18, metadata !DIExpression()), !dbg !23
  store i32 1, ptr %cleanup.dest.slot
  %4 = load ptr, ptr %saved_stack, !dbg !24
  call void @llvm.stackrestore(ptr %4), !dbg !24
  ret i32 0, !dbg !25
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare ptr @llvm.stacksave() nounwind

declare void @llvm.stackrestore(ptr) nounwind

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!27}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.1 (trunk 153698)", isOptimized: false, emissionKind: FullDebug, file: !26, enums: !1, retainedTypes: !1, globals: !1)
!1 = !{}
!5 = distinct !DISubprogram(name: "foo", line: 6, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, file: !26, scope: !0, type: !7)
!6 = !DIFile(filename: "20020104-2.c", directory: "/Volumes/Sandbox/llvm")
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !10}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !11)
!11 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !9)
!14 = !DILocalVariable(name: "x", line: 5, arg: 1, scope: !5, file: !6, type: !10)
!15 = !DILocation(line: 5, column: 21, scope: !5)
!16 = !DILocation(line: 7, column: 13, scope: !17)
!17 = distinct !DILexicalBlock(line: 6, column: 1, file: !26, scope: !5)
!18 = !DILocalVariable(name: "a", line: 7, scope: !17, file: !6, type: !19)
!19 = !DICompositeType(tag: DW_TAG_array_type, align: 8, baseType: !20, elements: !21)
!20 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!21 = !{!22}
!22 = !DISubrange(count: -1)
!23 = !DILocation(line: 7, column: 8, scope: !17)
!24 = !DILocation(line: 9, column: 1, scope: !17)
!25 = !DILocation(line: 8, column: 3, scope: !17)
!26 = !DIFile(filename: "20020104-2.c", directory: "/Volumes/Sandbox/llvm")
!27 = !{i32 1, !"Debug Info Version", i32 3}
