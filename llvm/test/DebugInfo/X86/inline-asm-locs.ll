; RUN: llc -mtriple=x86_64-apple-darwin %s -o - -filetype=asm | FileCheck %s

; Generated from:
; asm(
;       ".file 1 \"A.asm\"\n"
;       ".file 2 \"B.asm\"\n"
;       ".loc  1 111\n"
;       ".text\n"
;       ".globl _bar\n"
;       "_bar:\n"
;       ".loc 2 222\n"
;       "\tret\n"
;     );
;  
; void bar();
;  
; void foo() {
;   bar();
; }

; CHECK: .file 1 "A.asm"
; CHECK: .file 2 "B.asm"
; CHECK: .loc  1 111
; CHECK: .loc  2 222
; CHECK: .file 3 "test.c"
; CHECK: .loc  3 14 0  

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

module asm ".file 1 \22A.asm\22"
module asm ".file 2 \22B.asm\22"
module asm ".loc  1 111"
module asm ".text"
module asm ".globl _bar"
module asm "_bar:"
module asm ".loc 2 222"
module asm "\09ret"

; Function Attrs: nounwind ssp uwtable
define void @foo() !dbg !4 {
entry:
  call void (...) @bar(), !dbg !11
  ret void, !dbg !12
}

declare void @bar(...)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0 (trunk 256963)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, subprograms: !3)
!1 = !DIFile(filename: "test.c", directory: "/Volumes/Data/radar/22690666")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 14, type: !5, isLocal: false, isDefinition: true, scopeLine: 14, isOptimized: false, variables: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{i32 2, !"Dwarf Version", i32 2}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"PIC Level", i32 2}
!10 = !{!"clang version 3.8.0 (trunk 256963)"}
!11 = !DILocation(line: 15, column: 3, scope: !4)
!12 = !DILocation(line: 16, column: 1, scope: !4)
