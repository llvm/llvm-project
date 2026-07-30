; XFAIL: target={{.*}}-aix{{.*}}
; Tests the mangling escape prefix gets stripped from the linkage name.
;
; RUN: %llc_dwarf -accel-tables=Dwarf -dwarf-linkage-names=All -filetype=obj -o %t < %s
;
; RUN: llvm-dwarfdump -debug-info -debug-names %t | FileCheck %s
; RUN: llvm-dwarfdump -debug-names -verify %t | FileCheck --check-prefix=VERIFY %s

; CHECK: .debug_info contents:
; CHECK: DW_AT_linkage_name	("bar")
; CHECK: .debug_names contents:
; CHECK: String: {{.*}} "bar"

; VERIFY: No errors.

; Input generated from the following C++ code using
; clang -g -S -emit-llvm -target aarch64-apple-macos

; void foo() asm("bar");
; void foo() {}
; 
; void g() { foo(); }

define void @"\01bar"() !dbg !9 {
entry:
  ret void, !dbg !13
}

define void @_Z1gv() !dbg !14 {
entry:
  call void @"\01bar"(), !dbg !15
  ret void, !dbg !16
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 21.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
!1 = !DIFile(filename: "asm.cpp", directory: "/tmp", checksumkind: CSK_MD5, checksum: "d053f9249cc5548d446ceb58411ad625")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"uwtable", i32 1}
!7 = !{i32 7, !"frame-pointer", i32 1}
!8 = !{!"clang version 21.0.0git"}
!9 = distinct !DISubprogram(name: "foo", linkageName: "\01bar", scope: !10, file: !10, line: 2, type: !11, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!10 = !DIFile(filename: "asm.cpp", directory: "/tmp", checksumkind: CSK_MD5, checksum: "d053f9249cc5548d446ceb58411ad625")
!11 = !DISubroutineType(types: !12)
!12 = !{null}
!13 = !DILocation(line: 2, column: 13, scope: !9)
!14 = distinct !DISubprogram(name: "g", linkageName: "_Z1gv", scope: !10, file: !10, line: 4, type: !11, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!15 = !DILocation(line: 4, column: 12, scope: !14)
!16 = !DILocation(line: 4, column: 19, scope: !14)
