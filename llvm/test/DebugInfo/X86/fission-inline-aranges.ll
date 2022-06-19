; RUN: llc -generate-arange-section -split-dwarf-file=foo.dwo -O0 < %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj > %t
; RUN: llvm-dwarfdump -v -debug-aranges %t | FileCheck %s

; Test that only one entry is emmitted in .debug_aranges per CU.

; struct foo {
;    static void f2();
;    static void f3(...);
;  };
;
;  void foo::f3(...) {
;      f2();
; }

; Check that we emit only one entry in .debug_aranges

; CHECK:    cu_offset
; CHECK-NOT: cu_offset

; Function Attrs: mustprogress noinline optnone uwtable
define dso_local void @_ZN3foo2f3Ez(...) #0 align 2 !dbg !8 {
entry:
  call void @_ZN3foo2f2Ev(), !dbg !18
  ret void, !dbg !19
}

declare dso_local void @_ZN3foo2f2Ev() #1

attributes #0 = { mustprogress noinline optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, splitDebugFilename: "main.dwo", emissionKind: FullDebug, nameTableKind: GNU)
!1 = !DIFile(filename: "main.cpp", directory: "/tmp/dbginfo")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{i32 7, !"frame-pointer", i32 2}
!7 = !{!"clang version 14.0.0"}
!8 = distinct !DISubprogram(name: "f3", linkageName: "_ZN3foo2f3Ez", scope: !9, file: !1, line: 6, type: !15, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !14, retainedNodes: !17)
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !1, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !10, identifier: "_ZTS3foo")
!10 = !{!11, !14}
!11 = !DISubprogram(name: "f2", linkageName: "_ZN3foo2f2Ev", scope: !9, file: !1, line: 2, type: !12, scopeLine: 2, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!12 = !DISubroutineType(types: !13)
!13 = !{null}
!14 = !DISubprogram(name: "f3", linkageName: "_ZN3foo2f3Ez", scope: !9, file: !1, line: 3, type: !15, scopeLine: 3, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!15 = !DISubroutineType(types: !16)
!16 = !{null, null}
!17 = !{}
!18 = !DILocation(line: 7, column: 6, scope: !8)
!19 = !DILocation(line: 8, column: 2, scope: !8)
