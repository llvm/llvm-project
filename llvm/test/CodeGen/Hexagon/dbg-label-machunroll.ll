; RUN: llc -O3 -march=hexagon < %s | FileCheck %s
; CHECK-NOT: Segmentation

target triple = "hexagon-unknown-unknown-elf"

declare void @llvm.dbg.label(metadata)

define hidden fastcc i8 @__pyx_f_4lxml_5etree__decodeFilenameWithLength(i8* %__pyx_v_c_path, i32 %__pyx_v_c_len) unnamed_addr  {
entry:
  %add.ptr.i = getelementptr i8, i8* %__pyx_v_c_path, i32 1
  %0 = load i8, i8* %__pyx_v_c_path, align 1
  br label %while.cond.preheader.i

while.cond.preheader.i:                           ; preds = %entry
  %1 = load i8, i8* %add.ptr.i, align 1
  %2 = and i8 %1, -33
  %3 = add i8 %2, -65
  %4 = icmp ult i8 %3, 26
  br i1 %4, label %if.end101.i, label %if.end132.i

if.end101.i:                                      ; preds = %if.end101.i, %while.cond.preheader.i
  %__pyx_v_c_path.addr.0223.i = phi i8* [ %add.ptr102.i, %if.end101.i ], [ %add.ptr.i, %while.cond.preheader.i ]
  %add.ptr102.i = getelementptr i8, i8* %__pyx_v_c_path.addr.0223.i, i32 1
  %.pr.i = load i8, i8* %add.ptr102.i, align 1
  %5 = and i8 %.pr.i, -33
  %6 = add i8 %5, -65
  %7 = icmp ult i8 %6, 26
  call void @llvm.dbg.label(metadata !23), !dbg !27
  br i1 %7, label %if.end101.i, label %if.end132.i

if.end132.i:                                      ; preds = %while.cond.preheader.i
  ret i8 %0
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "QuIC LLVM Hexagon Clang version hexagon-clang-84-1613 (based on LLVM 9.0.0)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "lxml.etree.c", directory: "/local/mnt/workspace/santdas/src/llvm/master/qtool/qtool-42625")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"PIC Level", i32 2}
!9 = !DILabel(scope: !10, name: "__pyx_L5_bool_binop_done", file: !11, line: 32696)
!10 = distinct !DISubprogram(name: "__pyx_f_4lxml_5etree__isFilePath", scope: !11, file: !11, line: 32628, type: !12, scopeLine: 32628, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DIFile(filename: "src/lxml/lxml.etree.c", directory: "/local/mnt/workspace/santdas/src/llvm/master/qtool/qtool-42625")
!12 = !DISubroutineType(types: !13)
!13 = !{!14, !15}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 32)
!16 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !17)
!17 = !DIDerivedType(tag: DW_TAG_typedef, name: "xmlChar", file: !18, line: 28, baseType: !19)
!18 = !DIFile(filename: "scratch/buildroot/buildroot/output/host/usr/hexagon-buildroot-linux-musl/sysroot/usr/include/libxml2/libxml/xmlstring.h", directory: "/local/mnt/workspace")
!19 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!21 = distinct !DILocation(line: 33263, column: 16, scope: !22)
!22 = distinct !DISubprogram(name: "__pyx_f_4lxml_5etree__decodeFilenameWithLength", scope: !11, file: !11, line: 33236, type: !12, scopeLine: 33236, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!23 = !DILabel(scope: !24, name: "__pyx_L12_bool_binop_done", file: !11, line: 32782)
!24 = distinct !DILexicalBlock(scope: !25, file: !11, line: 32765, column: 15)
!25 = distinct !DILexicalBlock(scope: !26, file: !11, line: 32697, column: 18)
!26 = distinct !DILexicalBlock(scope: !10, file: !11, line: 32697, column: 7)
!27 = !DILocation(line: 32782, column: 7, scope: !24, inlinedAt: !21)