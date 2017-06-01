; RUN: opt < %s -S -inline -pass-remarks-missed=inline -pass-remarks-with-hotness \
; RUN:     -pass-remarks-output=%t 2>&1 | FileCheck %s
; RUN: cat %t | FileCheck -check-prefix=YAML %s
; RUN: opt < %s -S -inline -pass-remarks-with-hotness -pass-remarks-output=%t
; RUN: cat %t | FileCheck -check-prefix=YAML %s

; Check the YAML file generated for inliner remarks for this program:
;
;   1  int foo();
;   2  int bar();
;   3
;   4  int baz() {
;   5    return foo() + bar();
;   6  }

; CHECK:      remark: /tmp/s.c:5:10: foo will not be inlined into baz because its definition is unavailable (hotness: 30)
; CHECK-NEXT: remark: /tmp/s.c:5:18: bar will not be inlined into baz because its definition is unavailable (hotness: 30)

; YAML:      --- !Missed
; YAML-NEXT: Pass:            inline
; YAML-NEXT: Name:            NoDefinition
; YAML-NEXT: DebugLoc:        { File: /tmp/s.c, Line: 5, Column: 10 }
; YAML-NEXT: Function:        baz
; YAML-NEXT: Hotness:         30
; YAML-NEXT: Args:
; YAML-NEXT:   - Callee: foo
; YAML-NEXT:   - String: ' will not be inlined into '
; YAML-NEXT:   - Caller: baz
; YAML-NEXT:     DebugLoc:        { File: /tmp/s.c, Line: 4, Column: 0 }
; YAML-NEXT:   - String: ' because its definition is unavailable'
; YAML-NEXT: ...
; YAML-NEXT: --- !Missed
; YAML-NEXT: Pass:            inline
; YAML-NEXT: Name:            NoDefinition
; YAML-NEXT: DebugLoc:        { File: /tmp/s.c, Line: 5, Column: 18 }
; YAML-NEXT: Function:        baz
; YAML-NEXT: Hotness:         30
; YAML-NEXT: Args:
; YAML-NEXT:   - Callee: bar
; YAML-NEXT:   - String: ' will not be inlined into '
; YAML-NEXT:   - Caller: baz
; YAML-NEXT:     DebugLoc:        { File: /tmp/s.c, Line: 4, Column: 0 }
; YAML-NEXT:   - String: ' because its definition is unavailable'
; YAML-NEXT: ...

; ModuleID = '/tmp/s.c'
source_filename = "/tmp/s.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

; Function Attrs: nounwind ssp uwtable
define i32 @"\01baz"() !dbg !7 !prof !14 {
entry:
  %call = call i32 (...) @foo(), !dbg !9
  %call1 = call i32 (...) @"\01bar"(), !dbg !10
  %add = add nsw i32 %call, %call1, !dbg !12
  ret i32 %add, !dbg !13
}

declare i32 @foo(...)

declare i32 @"\01bar"(...)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0.0 (trunk 281293) (llvm/trunk 281290)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2)
!1 = !DIFile(filename: "/tmp/s.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 4.0.0 (trunk 281293) (llvm/trunk 281290)"}
!7 = distinct !DISubprogram(name: "baz", scope: !1, file: !1, line: 4, type: !8, isLocal: false, isDefinition: true, scopeLine: 4, isOptimized: true, unit: !0, variables: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 5, column: 10, scope: !7)
!10 = !DILocation(line: 5, column: 18, scope: !11)
!11 = !DILexicalBlockFile(scope: !7, file: !1, discriminator: 1)
!12 = !DILocation(line: 5, column: 16, scope: !7)
!13 = !DILocation(line: 5, column: 3, scope: !7)
!14 = !{!"function_entry_count", i64 30}
