; Check separated lines and ranges.
; RUN: opt < %s 2>&1 -disable-output -passes=forceattrs -print-after-all \
; RUN:   -filter-print-source-locs=source.c:10,30-32 \
; RUN:   | FileCheck %s --check-prefix=RANGE

; Check path suffix matching.
; RUN: opt < %s 2>&1 -disable-output -passes=forceattrs -print-after-all \
; RUN:   -filter-print-source-locs=dir/source.c:31 \
; RUN:   | FileCheck %s --check-prefix=SUFFIX

; Check function pass filtering.
; RUN: opt < %s 2>&1 -disable-output -passes='function(no-op-function)' \
; RUN:   -print-after-all -filter-print-source-locs=source.c:10 \
; RUN:   | FileCheck %s --check-prefix=FUNCTION

; Check that an explicit function wildcard still respects the source location
; filter.
; RUN: opt < %s 2>&1 -disable-output -passes=forceattrs -print-after-all \
; RUN:   -filter-print-funcs=* -filter-print-source-locs=source.c:10 \
; RUN:   | FileCheck %s --check-prefix=WILDCARD

; Check that a missing source location suppresses the dump.
; RUN: opt < %s 2>&1 -disable-output -passes=forceattrs -print-after-all \
; RUN:   -filter-print-source-locs=missing.c:10 \
; RUN:   | FileCheck %s --allow-empty --check-prefix=EMPTY

; Check that the explicit print pass uses the same filter.
; RUN: opt < %s 2>&1 -disable-output -passes=print \
; RUN:   -filter-print-funcs=* -filter-print-source-locs=source.c:10 \
; RUN:   | FileCheck %s --check-prefix=PRINT-PASS

; RANGE:      IR Dump After {{Force set function attributes|ForceFunctionAttrsPass}}
; RANGE:      define i32 @foo
; RANGE-NOT:  define i32 @bar
; RANGE:      define i32 @baz
; RANGE-NOT:  IR Dump After {{Force set function attributes|ForceFunctionAttrsPass}}

; SUFFIX:      IR Dump After {{Force set function attributes|ForceFunctionAttrsPass}}
; SUFFIX-NOT:  define i32 @foo
; SUFFIX-NOT:  define i32 @bar
; SUFFIX:      define i32 @baz
; SUFFIX-NOT:  IR Dump After {{Force set function attributes|ForceFunctionAttrsPass}}

; FUNCTION:      IR Dump After NoOpFunctionPass on foo
; FUNCTION-NEXT: define i32 @foo
; FUNCTION-NOT:  IR Dump After NoOpFunctionPass on bar
; FUNCTION-NOT:  IR Dump After NoOpFunctionPass on baz

; WILDCARD:      IR Dump After {{Force set function attributes|ForceFunctionAttrsPass}}
; WILDCARD:      define i32 @foo
; WILDCARD-NOT:  define i32 @bar
; WILDCARD-NOT:  define i32 @baz
; WILDCARD-NOT:  IR Dump After {{Force set function attributes|ForceFunctionAttrsPass}}

; EMPTY-NOT: IR Dump After {{Force set function attributes|ForceFunctionAttrsPass}}

; PRINT-PASS:      define i32 @foo
; PRINT-PASS-NOT:  define i32 @bar
; PRINT-PASS-NOT:  define i32 @baz

define i32 @foo() !dbg !5 {
entry:
  %sum = add i32 1, 2, !dbg !10
  ret i32 %sum, !dbg !11
}

define i32 @bar() !dbg !12 {
entry:
  %sum = add i32 3, 4, !dbg !13
  ret i32 %sum, !dbg !14
}

define i32 @baz() !dbg !15 {
entry:
  %sum = add i32 5, 6, !dbg !16
  ret i32 %sum, !dbg !17
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "source.c", directory: "/tmp/dir")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !18)
!5 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 9, type: !4, scopeLine: 9, spFlags: DISPFlagDefinition, unit: !0)
!10 = !DILocation(line: 10, column: 7, scope: !5)
!11 = !DILocation(line: 11, column: 3, scope: !5)
!12 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 19, type: !4, scopeLine: 19, spFlags: DISPFlagDefinition, unit: !0)
!13 = !DILocation(line: 20, column: 7, scope: !12)
!14 = !DILocation(line: 21, column: 3, scope: !12)
!15 = distinct !DISubprogram(name: "baz", scope: !1, file: !1, line: 29, type: !4, scopeLine: 29, spFlags: DISPFlagDefinition, unit: !0)
!16 = !DILocation(line: 31, column: 7, scope: !15)
!17 = !DILocation(line: 32, column: 3, scope: !15)
!18 = !{!19}
!19 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
