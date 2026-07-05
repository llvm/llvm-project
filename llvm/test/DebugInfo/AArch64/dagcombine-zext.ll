; RUN: llc -filetype=obj -o - %s | llvm-dwarfdump --name cntrl_flags - | FileCheck %s
; CHECK: DW_OP_reg0 W0
;
; void use(unsigned char);
; f(unsigned long long cntrl_flags,
;                           int page_count)
; {
;         unsigned char tag;
;         tag = (((cntrl_flags) >> 32) & 0xFF);
;         use(tag);
; }

source_filename = "test.i"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios5.0.0"

; Function Attrs: nounwind ssp uwtable
define i32 @f(i64 %cntrl_flags) local_unnamed_addr #0 !dbg !8 {
entry:
  tail call void @llvm.dbg.value(metadata i64 %cntrl_flags, metadata !14, metadata !DIExpression()), !dbg !18
  %shr = lshr i64 %cntrl_flags, 32, !dbg !20
  %conv = trunc i64 %shr to i8, !dbg !21
  tail call void @llvm.dbg.value(metadata i8 %conv, metadata !16, metadata !DIExpression()), !dbg !22
  tail call void @use(i8 zeroext %conv) #3, !dbg !23
  ret i32 undef, !dbg !24
}

declare void @use(i8 zeroext) local_unnamed_addr 

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind ssp uwtable }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { nobuiltin nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 (trunk 317700) (llvm/trunk 317708)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.i", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 6.0.0 (trunk 317700) (llvm/trunk 317708)"}
!8 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 2, type: !9, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !13)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!13 = !{!14, !16}
!14 = !DILocalVariable(name: "cntrl_flags", arg: 1, scope: !8, file: !1, line: 2, type: !12)
!16 = !DILocalVariable(name: "tag", scope: !8, file: !1, line: 5, type: !17)
!17 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!18 = !DILocation(line: 2, column: 22, scope: !8)
!19 = !DILocation(line: 3, column: 31, scope: !8)
!20 = !DILocation(line: 6, column: 24, scope: !8)
!21 = !DILocation(line: 6, column: 8, scope: !8)
!22 = !DILocation(line: 5, column: 16, scope: !8)
!23 = !DILocation(line: 7, column: 2, scope: !8)
!24 = !DILocation(line: 8, column: 1, scope: !8)
