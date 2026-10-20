; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu --dwarf-version=4 --basic-block-sections=none -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s
; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu --dwarf-version=4 --basic-block-sections=all -filetype=obj -o - | llvm-dwarfdump - | FileCheck --check-prefix=SECTIONS %s
; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu --dwarf-version=5 --basic-block-sections=none -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s
; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu --dwarf-version=5 --basic-block-sections=all -filetype=obj -o - | llvm-dwarfdump - | FileCheck --check-prefix=SECTIONS %s

; CHECK:      DW_TAG_lexical_block
; CHECK-NEXT: DW_AT_low_pc
; CHECK-NEXT: DW_AT_high_pc
; CHECK:      DW_TAG_variable
; CHECK-NEXT: DW_AT_const_value   (7)
; CHECK-NEXT: DW_AT_name  ("i")

; SECTIONS:      DW_TAG_lexical_block
; SECTIONS-NEXT: DW_AT_ranges
; SECTIONS:      DW_TAG_variable
; SECTIONS-NEXT: DW_AT_const_value   (7)
; SECTIONS-NEXT: DW_AT_name  ("i")   

; Test to check that a variable declared within a scope that has basic block
; sections still produces DW_AT_const_value.
; Source to generate the IR below:

; void f1(int *);
; extern bool b;
; int test() {
;     // i is const throughout the whole scope and should
;     // use DW_AT_const_value. The scope creates basic
;     // block sections and should use DW_AT_ranges.
;     int j = 10;
;     {
;       int i = 7;
;       f1(&j);
;       if (b)
;         f1(&j);
;     }
;     return j;
; }
;
; clang++ -S scoped_section_const.cc -g -O2 -emit-llvm

@b = external local_unnamed_addr global i8, align 1

; Function Attrs: mustprogress uwtable
define dso_local noundef i32 @_Z4testv() local_unnamed_addr #0 !dbg !9 {
  %1 = alloca i32, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %1) #4, !dbg !17
  call void @llvm.dbg.value(metadata i32 10, metadata !14, metadata !DIExpression()), !dbg !18
  store i32 10, ptr %1, align 4, !dbg !19, !tbaa !20
  call void @llvm.dbg.value(metadata i32 7, metadata !15, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata ptr %1, metadata !14, metadata !DIExpression(DW_OP_deref)), !dbg !18
  call void @_Z2f1Pi(ptr noundef nonnull %1), !dbg !25
  %2 = load i8, ptr @b, align 1, !dbg !26, !tbaa !28, !range !30, !noundef !31
  %3 = icmp eq i8 %2, 0, !dbg !26
  br i1 %3, label %5, label %4, !dbg !32

4:                                                ; preds = %0
  call void @llvm.dbg.value(metadata ptr %1, metadata !14, metadata !DIExpression(DW_OP_deref)), !dbg !18
  call void @_Z2f1Pi(ptr noundef nonnull %1), !dbg !33
  br label %5, !dbg !33

5:                                                ; preds = %4, %0
  %6 = load i32, ptr %1, align 4, !dbg !34, !tbaa !20
  call void @llvm.dbg.value(metadata i32 %6, metadata !14, metadata !DIExpression()), !dbg !18
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %1) #4, !dbg !35
  ret i32 %6, !dbg !36
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

declare !dbg !37 void @_Z2f1Pi(ptr noundef) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #0 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "Debian clang version 16.0.6 (26)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "scoped_section_const.cc", directory: "", checksumkind: CSK_MD5, checksum: "0406492d2e2e38af35d9ea210ba1f24b")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{!"Debian clang version 16.0.6 (26)"}
!9 = distinct !DISubprogram(name: "test", linkageName: "_Z4testv", scope: !1, file: !1, line: 3, type: !10, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{!14, !15}
!14 = !DILocalVariable(name: "j", scope: !9, file: !1, line: 6, type: !12)
!15 = !DILocalVariable(name: "i", scope: !16, file: !1, line: 8, type: !12)
!16 = distinct !DILexicalBlock(scope: !9, file: !1, line: 7, column: 5)
!17 = !DILocation(line: 6, column: 5, scope: !9)
!18 = !DILocation(line: 0, scope: !9)
!19 = !DILocation(line: 6, column: 9, scope: !9)
!20 = !{!21, !21, i64 0}
!21 = !{!"int", !22, i64 0}
!22 = !{!"omnipotent char", !23, i64 0}
!23 = !{!"Simple C++ TBAA"}
!24 = !DILocation(line: 0, scope: !16)
!25 = !DILocation(line: 9, column: 7, scope: !16)
!26 = !DILocation(line: 10, column: 11, scope: !27)
!27 = distinct !DILexicalBlock(scope: !16, file: !1, line: 10, column: 11)
!28 = !{!29, !29, i64 0}
!29 = !{!"bool", !22, i64 0}
!30 = !{i8 0, i8 2}
!31 = !{}
!32 = !DILocation(line: 10, column: 11, scope: !16)
!33 = !DILocation(line: 11, column: 9, scope: !27)
!34 = !DILocation(line: 13, column: 12, scope: !9)
!35 = !DILocation(line: 14, column: 1, scope: !9)
!36 = !DILocation(line: 13, column: 5, scope: !9)
!37 = !DISubprogram(name: "f1", linkageName: "_Z2f1Pi", scope: !1, file: !1, line: 1, type: !38, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !31)
!38 = !DISubroutineType(types: !39)
!39 = !{null, !40}
!40 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
