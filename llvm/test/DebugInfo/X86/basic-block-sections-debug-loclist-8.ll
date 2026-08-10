; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu --dwarf-version=4 --basic-block-sections=none -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s
; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu --dwarf-version=4 --basic-block-sections=all -filetype=obj -o - | llvm-dwarfdump - | FileCheck  %s
; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu --dwarf-version=5 --basic-block-sections=none -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s
; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu --dwarf-version=5 --basic-block-sections=all -filetype=obj -o - | llvm-dwarfdump - | FileCheck  %s

; CHECK:      DW_TAG_variable
; CHECK:      DW_TAG_variable
; CHECK-NEXT: DW_AT_location
; CHECK-NEXT: DW_OP_consts +0, DW_OP_stack_value
; CHECK-NEXT: DW_OP_consts +7, DW_OP_stack_value
; CHECK-NEXT: DW_OP_consts +8, DW_OP_stack_value
; CHECK: DW_AT_name  ("i")

; void f1(int *);
; void f2(int);
; extern bool b;
; int test() {
;     // i is not a const throughout the whole scope and
;     // should *not* use DW_AT_const_value.
;     int i = 0;
;     int j = 10;
;     {
;       i = 7;
;       f1(&j);
;    }
;     i = 8;
;     f2(i);
;     return j;
; }
; clang++ -S scoped_section.cc -g -O2 -emit-llvm

; Function Attrs: mustprogress uwtable
define dso_local noundef i32 @_Z4testv() local_unnamed_addr #0 !dbg !10 {
entry:
  %j = alloca i32, align 4, !DIAssignID !17
    #dbg_assign(i1 undef, !16, !DIExpression(), !17, ptr %j, !DIExpression(), !18)
    #dbg_value(i32 0, !15, !DIExpression(), !18)
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %j) #3, !dbg !19
  store i32 10, ptr %j, align 4, !dbg !20, !tbaa !21, !DIAssignID !25
    #dbg_assign(i32 10, !16, !DIExpression(), !25, ptr %j, !DIExpression(), !18)
    #dbg_value(i32 7, !15, !DIExpression(), !18)
  call void @_Z2f1Pi(ptr noundef nonnull %j), !dbg !26
    #dbg_value(i32 8, !15, !DIExpression(), !18)
  call void @_Z2f2i(i32 noundef 8), !dbg !28
  %0 = load i32, ptr %j, align 4, !dbg !29, !tbaa !21
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %j) #3, !dbg !30
  ret i32 %0, !dbg !31
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

declare !dbg !32 void @_Z2f1Pi(ptr noundef) local_unnamed_addr #2

declare !dbg !36 void @_Z2f2i(i32 noundef) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

attributes #0 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 20.0.0git (git@github.com:tmsri/llvm-project.git 11a50269e82b6dce49249c5cbe3a989b06f0848f)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "scoped_section.cc", directory: "", checksumkind: CSK_MD5, checksum: "2d5675e292541e4f04eb60edf76b14d6")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang version 20.0.0git (git@github.com:tmsri/llvm-project.git 11a50269e82b6dce49249c5cbe3a989b06f0848f)"}
!10 = distinct !DISubprogram(name: "test", linkageName: "_Z4testv", scope: !1, file: !1, line: 4, type: !11, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{!15, !16}
!15 = !DILocalVariable(name: "i", scope: !10, file: !1, line: 7, type: !13)
!16 = !DILocalVariable(name: "j", scope: !10, file: !1, line: 8, type: !13)
!17 = distinct !DIAssignID()
!18 = !DILocation(line: 0, scope: !10)
!19 = !DILocation(line: 8, column: 5, scope: !10)
!20 = !DILocation(line: 8, column: 9, scope: !10)
!21 = !{!22, !22, i64 0}
!22 = !{!"int", !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C++ TBAA"}
!25 = distinct !DIAssignID()
!26 = !DILocation(line: 11, column: 7, scope: !27)
!27 = distinct !DILexicalBlock(scope: !10, file: !1, line: 9, column: 5)
!28 = !DILocation(line: 14, column: 5, scope: !10)
!29 = !DILocation(line: 15, column: 12, scope: !10)
!30 = !DILocation(line: 16, column: 1, scope: !10)
!31 = !DILocation(line: 15, column: 5, scope: !10)
!32 = !DISubprogram(name: "f1", linkageName: "_Z2f1Pi", scope: !1, file: !1, line: 1, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!33 = !DISubroutineType(types: !34)
!34 = !{null, !35}
!35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!36 = !DISubprogram(name: "f2", linkageName: "_Z2f2i", scope: !1, file: !1, line: 2, type: !37, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!37 = !DISubroutineType(types: !38)
!38 = !{null, !13}