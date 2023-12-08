; Check that we handle template types for split-dwarf-inlining correctly.
; RUN: llc -split-dwarf-file=%t.dwo -O2 < %s -dwarf-version=5 -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o -  | llvm-dwarfdump - | FileCheck %s
;
; The test case is generated from the following code
; clang -cc1 -emit-llvm -fdebug-info-for-profiling -fsplit-dwarf-inlining -debug-info-kind=constructor -dwarf-version=5 -split-dwarf-file temp.dwo -O2
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
; CHECK-NEXT:     DW_AT_name	("f2<int>")
; CHECK:          DW_TAG_template_type_parameter
; CHECK-NEXT:       DW_AT_type	(0x{{.*}} "int")
; CHECK-NEXT:       DW_AT_name	("T")
; CHECK:      .debug_info.dwo contents:
; CHECK:        DW_TAG_compile_unit
; CHECK:          DW_TAG_subprogram
; CHECK-NEXT:     DW_AT_linkage_name	("_Z2f2IiEvv")
; CHECK-NEXT:     DW_AT_name	("f2<int>")
; CHECK:          DW_TAG_template_type_parameter
; CHECK-NEXT:       DW_AT_type	(0x{{.*}} "int")
; CHECK-NEXT:       DW_AT_name	("T")

; ModuleID = 'split-debug-inlining-template.cpp'
source_filename = "llvm-project/clang/test/CodeGen/split-debug-inlining-template.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-redhat-linux-gnu"

; Function Attrs: mustprogress nounwind
define dso_local void @_Z2f3v() local_unnamed_addr #0 !dbg !6 {
entry:
  tail call void @_Z2f1v() #2, !dbg !11
  ret void, !dbg !17
}

declare !dbg !18 void @_Z2f1v() local_unnamed_addr #1

attributes #0 = { mustprogress nounwind "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #1 = { "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 15.0.0 (https://github.com/llvm/llvm-project.git 8b2b9c3fa91ebe583f8c634482885a669b82a1f0)", isOptimized: true, runtimeVersion: 0, splitDebugFilename: "split-debug-inlining-template.c.tmp.dwo", emissionKind: FullDebug, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "llvm-project/clang/test/CodeGen/<stdin>", directory: "build-debug", checksumkind: CSK_MD5, checksum: "0fb39b3bb5a60928b5d9c251b2d91b2c")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project.git 8b2b9c3fa91ebe583f8c634482885a669b82a1f0)"}
!6 = distinct !DISubprogram(name: "f3", linkageName: "_Z2f3v", scope: !7, file: !7, line: 11, type: !8, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!7 = !DIFile(filename: "llvm-project/clang/test/CodeGen/split-debug-inlining-template.cpp", directory: "", checksumkind: CSK_MD5, checksum: "0fb39b3bb5a60928b5d9c251b2d91b2c")
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{}
!11 = !DILocation(line: 8, column: 3, scope: !12, inlinedAt: !16)
!12 = distinct !DISubprogram(name: "f2<int>", linkageName: "_Z2f2IiEvv", scope: !7, file: !7, line: 7, type: !8, scopeLine: 7, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, templateParams: !13, retainedNodes: !10)
!13 = !{!14}
!14 = !DITemplateTypeParameter(name: "T", type: !15)
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = distinct !DILocation(line: 12, column: 3, scope: !6)
!17 = !DILocation(line: 13, column: 1, scope: !6)
!18 = !DISubprogram(name: "f1", linkageName: "_Z2f1v", scope: !7, file: !7, line: 4, type: !8, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !10)
