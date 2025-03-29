; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: llc -filetype=obj -o - a.ll -split-dwarf-file=foo.dwo \
; RUN:     | llvm-dwarfdump - -debug-names -debug-info \
; RUN:     | FileCheck --implicit-check-not=contents: %s

; CHECK: .debug_info contents:
; CHECK: .debug_info.dwo contents:
; CHECK: DW_TAG_subprogram
; CHECK: [[F3_DEF:0x[0-9a-f]*]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name ("f3")
; CHECK-NOT: DW_TAG
; CHECK: [[F2_INL:0x[0-9a-f]*]]:   DW_TAG_inlined_subroutine
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_abstract_origin ({{.*}} "_Z2f2v")
; CHECK: .debug_names contents:
; CHECK: String: {{.*}} "f2"
; CHECK: Entry {{.*}}{
; CHECK-NOT: {{^ *}}Entry
; CHECK:   Tag: DW_TAG_inlined_subroutine
; CHECK-NOT: {{^ *}}Entry
; CHECK:   DW_IDX_die_offset: [[F2_INL]]
; CHECK-NOT: {{^ *}}Entry
; CHECK: }
; CHECK-NEXT: }
; CHECK: String: {{.*}} "f3"
; CHECK: Entry {{.*}}{
; CHECK-NOT: {{^ *}}Entry
; CHECK:   Tag: DW_TAG_subprogram
; CHECK-NOT: {{^ *}}Entry
; CHECK:   DW_IDX_die_offset: [[F3_DEF]]
; CHECK-NOT: {{^ *}}Entry
; CHECK: }
; CHECK-NEXT: }

;--- a.cc
void f1();
inline void f2() {
  f1();
}
void f3() {
  f2();
}
;--- gen
clang++ --target=x86_64-linux -S -emit-llvm -O3 -g -fsplit-dwarf-inlining -c \
		 -gpubnames a.cc -o -

;--- a.ll
; ModuleID = 'a.cc'
source_filename = "a.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux"

; Function Attrs: mustprogress uwtable
define dso_local void @_Z2f3v() local_unnamed_addr #0 !dbg !8 {
entry:
  tail call void @_Z2f1v(), !dbg !11
  ret void, !dbg !14
}

declare !dbg !15 void @_Z2f1v() local_unnamed_addr #1

attributes #0 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "a.cc", directory: "/proc/self/cwd", checksumkind: CSK_MD5, checksum: "661bc2c3d7df6cc69861afc8fb17bf16")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = distinct !DISubprogram(name: "f3", linkageName: "_Z2f3v", scope: !1, file: !1, line: 5, type: !9, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DILocation(line: 3, column: 3, scope: !12, inlinedAt: !13)
!12 = distinct !DISubprogram(name: "f2", linkageName: "_Z2f2v", scope: !1, file: !1, line: 2, type: !9, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!13 = distinct !DILocation(line: 6, column: 3, scope: !8)
!14 = !DILocation(line: 7, column: 1, scope: !8)
!15 = !DISubprogram(name: "f1", linkageName: "_Z2f1v", scope: !1, file: !1, line: 1, type: !9, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
