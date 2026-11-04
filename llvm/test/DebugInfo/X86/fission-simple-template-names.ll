; Check that we handle template types for split-dwarf-inlining and simple-template-names correctly.
; RUN: llc -split-dwarf-file=%t.dwo -O2 < %s -dwarf-version=5 -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o -  | llvm-dwarfdump - | FileCheck %s
;
; The test case is generated from the following code
; clang -cc1 -emit-llvm -fdebug-info-for-profiling -fsplit-dwarf-inlining -gsimple-template-names=simple -debug-info-kind=constructor -dwarf-version=5 -split-dwarf-file temp.dwo -O2
;
; void f1();
;
; template <typename T>
; void f2() {
;   f1();
; }
;
; void f3() {
;   f2<int>();
; }

; CHECK:      .debug_info contents:
; CHECK:        DW_TAG_skeleton_unit
; CHECK:          DW_TAG_subprogram
; CHECK-NEXT:     DW_AT_linkage_name	("_Z2f2IiEvv")
; CHECK-NEXT:     DW_AT_name	("f2")
; CHECK:          DW_TAG_template_type_parameter
; CHECK-NEXT:       DW_AT_type	(0x{{.*}} "int")
; CHECK-NEXT:       DW_AT_name	("T")
; CHECK:      .debug_info.dwo contents:
; CHECK:        DW_TAG_compile_unit
; CHECK:          DW_TAG_subprogram
; CHECK-NEXT:     DW_AT_linkage_name	("_Z2f2IiEvv")
; CHECK-NEXT:     DW_AT_name	("f2")
; CHECK:          DW_TAG_template_type_parameter
; CHECK-NEXT:       DW_AT_type	(0x{{.*}} "int")
; CHECK-NEXT:       DW_AT_name	("T")

; ModuleID = 'fission-simple-template-names.cpp'
source_filename = "fission-simple-template-names.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress nounwind
define dso_local void @_Z2f3v() local_unnamed_addr #0 !dbg !10 {
entry:
  tail call void @_Z2f1v() #2, !dbg !14
  ret void, !dbg !20
}

declare !dbg !21 void @_Z2f1v() local_unnamed_addr #1

attributes #0 = { mustprogress nounwind "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}
!llvm.ident = !{!5}
!llvm.errno.tbaa = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 22.0.0", isOptimized: true, runtimeVersion: 0, splitDebugFilename: "temp.dwo", emissionKind: FullDebug, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "", checksumkind: CSK_MD5, checksum: "f34ede93f4bfefbee3dbe78d1a033274")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 22.0.0"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = distinct !DISubprogram(name: "f3", linkageName: "_Z2f3v", scope: !11, file: !11, line: 8, type: !12, scopeLine: 8, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DIFile(filename: "fission-simple-template-names.cpp", directory: "", checksumkind: CSK_MD5, checksum: "f34ede93f4bfefbee3dbe78d1a033274")
!12 = !DISubroutineType(types: !13)
!13 = !{null}
!14 = !DILocation(line: 5, column: 3, scope: !15, inlinedAt: !19)
!15 = distinct !DISubprogram(name: "f2", linkageName: "_Z2f2IiEvv", scope: !11, file: !11, line: 4, type: !12, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed | DIFlagNameIsSimplified, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, templateParams: !16)
!16 = !{!17}
!17 = !DITemplateTypeParameter(name: "T", type: !18)
!18 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!19 = distinct !DILocation(line: 9, column: 3, scope: !10)
!20 = !DILocation(line: 10, column: 1, scope: !10)
!21 = !DISubprogram(name: "f1", linkageName: "_Z2f1v", scope: !11, file: !11, line: 1, type: !12, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
