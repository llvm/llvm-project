;; The dbg value for buflen in the non-entry basic block spans the entire
;; function and is emitted as DW_AT_const_value.  Even with basic block
;; sections, this can be done as the entire function is represented as ranges.

; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu --dwarf-version=4 --basic-block-sections=none -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s
; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu --dwarf-version=4 --basic-block-sections=all -filetype=obj -o -  | llvm-dwarfdump - | FileCheck %s
; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu --dwarf-version=5 --basic-block-sections=none -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s
; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu --dwarf-version=5 --basic-block-sections=all -filetype=obj -o -  | llvm-dwarfdump - | FileCheck %s

; CHECK:      DW_AT_const_value (157)
; CHECK-NEXT: DW_AT_name ("buflen")

;; We do not have the source to reproduce this as this was IR was obtained
;; using a reducer from a failing compile.

define dso_local void @_ZL4ncatPcjz(ptr %0, i32 %1, ...) unnamed_addr  align 32 !dbg !22 {
.critedge3:
  call void @llvm.dbg.value(metadata i32 157, metadata !27, metadata !DIExpression()), !dbg !46
  call void @llvm.va_start(ptr nonnull undef), !dbg !47
  br label %2

2:                                                ; preds = %2, %.critedge3
  br label %2
}

declare void @llvm.va_start(ptr)

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!20, !21}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2)
!1 = !DIFile(filename: "bug.cpp", directory: "/proc/self/cwd")
!2 = !{!3, !6, !9, !10, !11, !13, !16, !17, !15}
!3 = !DIDerivedType(tag: DW_TAG_typedef, name: "int32_t", file: !4, line: 38, baseType: !5)
!4 = !DIFile(filename: "stdint.h", directory: "/proc/self/cwd")
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!7 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !8)
!8 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_unsigned_char)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!11 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint32_t", file: !4, line: 51, baseType: !12)
!12 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!13 = !DIDerivedType(tag: DW_TAG_typedef, name: "UChar", file: !14, line: 372, baseType: !15)
!14 = !DIFile(filename: "umachine.h", directory: "/proc/self/cwd")
!15 = !DIBasicType(name: "char16_t", size: 16, encoding: DW_ATE_UTF)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64)
!17 = !DIDerivedType(tag: DW_TAG_typedef, name: "UBool", file: !14, line: 261, baseType: !18)
!18 = !DIDerivedType(tag: DW_TAG_typedef, name: "int8_t", file: !4, line: 36, baseType: !19)
!19 = !DIBasicType(name: "signed char", size: 8, encoding: DW_ATE_signed_char)
!20 = !{i32 2, !"Debug Info Version", i32 3}
!21 = !{i32 7, !"PIC Level", i32 2}
!22 = distinct !DISubprogram(name: "ncat", linkageName: "_ZL4ncatPcjz", scope: !1, file: !1, line: 37, type: !23, scopeLine: 37, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !25)
!23 = !DISubroutineType(types: !24)
!24 = !{!3, !9, !11, null}
!25 = !{!26, !27, !28, !41, !42, !43, !44}
!26 = !DILocalVariable(name: "buffer", arg: 1, scope: !22, file: !1, line: 37, type: !9)
!27 = !DILocalVariable(name: "buflen", arg: 2, scope: !22, file: !1, line: 37, type: !11)
!28 = !DILocalVariable(name: "args", scope: !22, file: !1, line: 38, type: !29)
!29 = !DIDerivedType(tag: DW_TAG_typedef, name: "va_list", file: !30, line: 14, baseType: !31)
!30 = !DIFile(filename: "stdarg.h", directory: "/proc/self/cwd")
!31 = !DIDerivedType(tag: DW_TAG_typedef, name: "__builtin_va_list", file: !1, line: 38, baseType: !32)
!32 = !DICompositeType(tag: DW_TAG_array_type, baseType: !33, size: 192, elements: !39)
!33 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__va_list_tag", file: !1, line: 38, size: 192, flags: DIFlagTypePassByValue, elements: !34, identifier: "_ZTS13__va_list_tag")
!34 = !{!35, !36, !37, !38}
!35 = !DIDerivedType(tag: DW_TAG_member, name: "gp_offset", scope: !33, file: !1, line: 38, baseType: !12, size: 32)
!36 = !DIDerivedType(tag: DW_TAG_member, name: "fp_offset", scope: !33, file: !1, line: 38, baseType: !12, size: 32, offset: 32)
!37 = !DIDerivedType(tag: DW_TAG_member, name: "overflow_arg_area", scope: !33, file: !1, line: 38, baseType: !10, size: 64, offset: 64)
!38 = !DIDerivedType(tag: DW_TAG_member, name: "reg_save_area", scope: !33, file: !1, line: 38, baseType: !10, size: 64, offset: 128)
!39 = !{!40}
!40 = !DISubrange(count: 1)
!41 = !DILocalVariable(name: "str", scope: !22, file: !1, line: 39, type: !9)
!42 = !DILocalVariable(name: "p", scope: !22, file: !1, line: 40, type: !9)
!43 = !DILocalVariable(name: "e", scope: !22, file: !1, line: 41, type: !6)
!44 = !DILocalVariable(name: "c", scope: !45, file: !1, line: 49, type: !8)
!45 = distinct !DILexicalBlock(scope: !22, file: !1, line: 48, column: 45)
!46 = !DILocation(line: 0, scope: !22)
!47 = !DILocation(line: 47, column: 3, scope: !22)
