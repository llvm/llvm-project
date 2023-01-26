; RUN: llvm-link %s %S/Inputs/subprogram-linkonce-weak.ll -S -o %t1
; RUN: FileCheck %s -check-prefix=LW -check-prefix=CHECK <%t1
; RUN: llvm-link %S/Inputs/subprogram-linkonce-weak.ll %s -S -o %t2
; RUN: FileCheck %s -check-prefix=WL -check-prefix=CHECK <%t2
; REQUIRES: object-emission
;
; Bug 47131
; XFAIL: target=sparc{{.*}}
;
; This testcase tests the following flow:
;  - File A defines a linkonce version of @foo which has inlined into @bar.
;  - File B defines a weak version of @foo (different definition).
;  - Linkage rules state File B version of @foo wins.
;  - @bar still has inlined debug info related to the linkonce @foo.
;
; This should fix PR22792, although the testcase was hand-written.  There's a
; RUN line with a crasher for llc at the end with checks for the DWARF output.

; The LW prefix means linkonce (this file) first, then weak (the other file).
; The WL prefix means weak (the other file) first, then linkonce (this file).

; We'll see @bar before @foo if this file is first.
; LW: define i32 @bar({{.*}} !dbg ![[BARSP:[0-9]+]]
; LW: %sum = add i32 %a, %b, !dbg ![[FOOINBAR:[0-9]+]]
; LW: ret i32 %sum, !dbg ![[BARRET:[0-9]+]]
; LW: define weak i32 @foo({{.*}} !dbg ![[WEAKFOOSP:[0-9]+]]
; LW: %sum = call i32 @fastadd(i32 %a, i32 %b), !dbg ![[FOOCALL:[0-9]+]]
; LW: ret i32 %sum, !dbg ![[FOORET:[0-9]+]]

; We'll see @foo before @bar if this file is second.
; WL: define weak i32 @foo({{.*}} !dbg ![[WEAKFOOSP:[0-9]+]]
; WL: %sum = call i32 @fastadd(i32 %a, i32 %b), !dbg ![[FOOCALL:[0-9]+]]
; WL: ret i32 %sum, !dbg ![[FOORET:[0-9]+]]
; WL: define i32 @bar({{.*}} !dbg ![[BARSP:[0-9]+]]
; WL: %sum = add i32 %a, %b, !dbg ![[FOOINBAR:[0-9]+]]
; WL: ret i32 %sum, !dbg ![[BARRET:[0-9]+]]

define i32 @bar(i32 %a, i32 %b) !dbg !3 {
entry:
  %sum = add i32 %a, %b, !dbg !DILocation(line: 2, scope: !4,
                                          inlinedAt: !DILocation(line: 12, scope: !3))
  ret i32 %sum, !dbg !DILocation(line: 13, scope: !3)
}

define linkonce i32 @foo(i32 %a, i32 %b) !dbg !4 {
entry:
  %sum = add i32 %a, %b, !dbg !DILocation(line: 2, scope: !4)
  ret i32 %sum, !dbg !DILocation(line: 3, scope: !4)
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}

; CHECK-LABEL: !llvm.dbg.cu =
; LW-SAME: !{![[LCU:[0-9]+]], ![[WCU:[0-9]+]]}
; WL-SAME: !{![[WCU:[0-9]+]], ![[LCU:[0-9]+]]}
!llvm.dbg.cu = !{!1}

; LW: ![[LCU]] = distinct !DICompileUnit(
; LW: ![[WCU]] = distinct !DICompileUnit(
; LW: ![[BARSP]] = distinct !DISubprogram(name: "bar",{{.*}} unit: ![[LCU]]
; LW: ![[FOOINBAR]] = !DILocation(line: 2, scope: ![[FOOSP:.*]], inlinedAt: ![[BARIA:[0-9]+]])
; LW: ![[FOOSP]] = distinct !DISubprogram(name: "foo",{{.*}} unit: ![[LCU]]
; LW: ![[BARIA]] = !DILocation(line: 12, scope: ![[BARSP]])
; LW: ![[BARRET]] = !DILocation(line: 13, scope: ![[BARSP]])
; LW: ![[WEAKFOOSP]] = distinct !DISubprogram(name: "foo",{{.*}} unit: ![[WCU]]
; LW: ![[FOOCALL]] = !DILocation(line: 52, scope: ![[WEAKFOOSP]])
; LW: ![[FOORET]] = !DILocation(line: 53, scope: ![[WEAKFOOSP]])

; Same as above, but reordered.
; WL: ![[WCU]] = distinct !DICompileUnit(
; WL: ![[LCU]] = distinct !DICompileUnit(
; WL: ![[WEAKFOOSP]] = distinct !DISubprogram(name: "foo",{{.*}} unit: ![[WCU]]
; WL: ![[FOOCALL]] = !DILocation(line: 52, scope: ![[WEAKFOOSP]])
; WL: ![[FOORET]] = !DILocation(line: 53, scope: ![[WEAKFOOSP]])
; WL: ![[BARSP]] = distinct !DISubprogram(name: "bar",{{.*}} unit: ![[LCU]]
; WL: ![[FOOINBAR]] = !DILocation(line: 2, scope: ![[FOOSP:.*]], inlinedAt: ![[BARIA:[0-9]+]])
; WL: ![[FOOSP]] = distinct !DISubprogram(name: "foo",{{.*}} unit: ![[LCU]]
; WL: ![[BARIA]] = !DILocation(line: 12, scope: ![[BARSP]])
; WL: ![[BARRET]] = !DILocation(line: 13, scope: ![[BARSP]])

!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, emissionKind: FullDebug)
!2 = !DIFile(filename: "bar.c", directory: "/path/to/dir")
!3 = distinct !DISubprogram(file: !2, scope: !2, line: 11, name: "bar", type: !5, unit: !1)
!4 = distinct !DISubprogram(file: !2, scope: !2, line: 1, name: "foo", type: !5, unit: !1)
!5 = !DISubroutineType(types: !{})

; Crasher for llc.
; RUN: %llc_dwarf -filetype=obj -O0 %t1 -o %t1.o
; RUN: llvm-dwarfdump %t1.o --all | FileCheck %s -check-prefix=DWLW -check-prefix=DW
; RUN: %llc_dwarf -filetype=obj -O0 %t2 -o %t2.o
; RUN: llvm-dwarfdump %t2.o --all | FileCheck %s -check-prefix=DWWL -check-prefix=DW
; Check that the debug info for the discarded linkonce version of @foo doesn't
; reference any code, and that the other subprograms look correct.

; DW-LABEL: .debug_info contents:
; DWLW:     DW_TAG_compile_unit
; DWLW:       DW_AT_name {{.*}}"bar.c"
; DWLW:       DW_TAG_subprogram
; DWLW-NOT:     DW_AT_low_pc
; DWLW-NOT:     DW_AT_high_pc
; DWLW:         DW_AT_name {{.*}}foo
; DWLW:         DW_AT_decl_file {{.*}}"/path/to/dir{{/|\\}}bar.c"
; DWLW:         DW_AT_decl_line {{.*}}(1)
; DWLW:       DW_TAG_subprogram
; DWLW:         DW_AT_low_pc
; DWLW:         DW_AT_high_pc
; DWLW:         DW_AT_name {{.*}}bar
; DWLW:         DW_AT_decl_file {{.*}}"/path/to/dir{{/|\\}}bar.c"
; DWLW:         DW_AT_decl_line {{.*}}(11)

; DWLW:         DW_TAG_inlined_subroutine
; DWLW:           DW_AT_abstract_origin
; DWLW:     DW_TAG_compile_unit
; DWLW:       DW_AT_name {{.*}}"foo.c"
; DWLW:       DW_TAG_subprogram
; DWLW:         DW_AT_low_pc
; DWLW:         DW_AT_high_pc
; DWLW:         DW_AT_name {{.*}}foo
; DWLW:         DW_AT_decl_file {{.*}}"/path/to/dir{{/|\\}}foo.c"
; DWLW:         DW_AT_decl_line {{.*}}(51)

; The DWARF output is already symmetric (just reordered).
; DWWL:     DW_TAG_compile_unit
; DWWL:       DW_AT_name {{.*}}"foo.c"
; DWWL:       DW_TAG_subprogram
; DWWL:         DW_AT_low_pc
; DWWL:         DW_AT_high_pc
; DWWL:         DW_AT_name {{.*}}foo
; DWWL:         DW_AT_decl_file {{.*}}"/path/to/dir{{/|\\}}foo.c"
; DWWL:         DW_AT_decl_line {{.*}}(51)
; DWWL:     DW_TAG_compile_unit
; DWWL:       DW_AT_name {{.*}}"bar.c"
; DWWL:       DW_TAG_subprogram
; DWWL-NOT:     DW_AT_low_pc
; DWWL-NOT:     DW_AT_high_pc
; DWWL:         DW_AT_name {{.*}}foo
; DWWL:         DW_AT_decl_file {{.*}}"/path/to/dir{{/|\\}}bar.c"
; DWWL:         DW_AT_decl_line {{.*}}(1)
; DWWL:       DW_TAG_subprogram
; DWWL:         DW_AT_low_pc
; DWWL:         DW_AT_high_pc
; DWWL:         DW_AT_name {{.*}}bar
; DWWL:         DW_AT_decl_file {{.*}}"/path/to/dir{{/|\\}}bar.c"
; DWWL:         DW_AT_decl_line {{.*}}(11)
; DWWL:         DW_TAG_inlined_subroutine
; DWWL:           DW_AT_abstract_origin

; DW-LABEL:   .debug_line contents:
; Check that we have the right things in the line table as well.

; DWLW-LABEL: file_names[ 1]:
; DWLW-NEXT: name: "bar.c"
; DWLW:        2 0 1 0 0 is_stmt prologue_end
; DWLW-LABEL: file_names[ 1]:
; DWLW-NEXT: name: "foo.c"
; DWLW:       52 0 1 0 0 is_stmt prologue_end
; DWLW-NOT:                      prologue_end

; DWWL-LABEL: file_names[ 1]:
; DWWL-NEXT: name: "foo.c"
; DWWL:       52 0 1 0 0 is_stmt prologue_end
; DWWL-LABEL: file_names[ 1]:
; DWWL-NEXT: name: "bar.c"
; DWWL:        2 0 1 0 0 is_stmt prologue_end
; DWWL-NOT:                      prologue_end
