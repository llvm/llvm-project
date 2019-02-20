; RUN: llc -dwarf-version=5 -O0 %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump -v %t | FileCheck %s
;
; Check that we generate DW_AT_rnglists_base in the CU die, as well as a range 
; list table when we have more than 1 CU range and no scope range.
;
; Generated from:
; 
; __attribute__((section("text.foo"))) void f1() {}
; __attribute__((section("text.bar"))) void f2() {}
;
; Compile with clangc -gdwarf-5 -O0 -S -emit-llvm
;
; CHECK:      .debug_info contents:
; CHECK:      DW_TAG_compile_unit
; CHECK-NOT:  DW_TAG
; CHECK:      DW_AT_rnglists_base [DW_FORM_sec_offset]                   (0x0000000c)
; CHECK:      .debug_rnglists contents:
; CHECK:      0x00000000: range list header: length = 0x00000013, version = 0x0005,
; CHECK-SAME: addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000001

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @f1() section "text.foo" !dbg !7 {
entry:
  ret void, !dbg !10
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @f2() section "text.bar" !dbg !11 {
entry:
  ret void, !dbg !12
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 7.0.0 (trunk 337470)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test3.c", directory: "/home/test/rangelists", checksumkind: CSK_MD5, checksum: "f3b46bc2e5bc55bdd511ae4ec29577b6")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 7.0.0 (trunk 337470)"}
!7 = distinct !DISubprogram(name: "f1", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocation(line: 1, column: 49, scope: !7)
!11 = distinct !DISubprogram(name: "f2", scope: !1, file: !1, line: 2, type: !8, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: false, unit: !0, retainedNodes: !2)
!12 = !DILocation(line: 2, column: 49, scope: !11)
