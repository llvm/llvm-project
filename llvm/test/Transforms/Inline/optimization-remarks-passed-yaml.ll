; RUN: opt < %s -S -inline -pass-remarks-output=%t -pass-remarks=inline \
; RUN:    -pass-remarks-missed=inline -pass-remarks-analysis=inline \
; RUN:    -pass-remarks-with-hotness 2>&1 | FileCheck %s
; RUN: cat %t | FileCheck -check-prefixes=YAML,YAML-NO-ANNOTATE %s

; RUN: opt < %s -S -passes=inline -pass-remarks-output=%t -pass-remarks=inline \
; RUN:    -pass-remarks-missed=inline -pass-remarks-analysis=inline \
; RUN:    -pass-remarks-with-hotness 2>&1 | FileCheck %s
; RUN: cat %t | FileCheck -check-prefixes=YAML,YAML-NO-ANNOTATE %s

; RUN: opt < %s -S -passes=inliner-wrapper -pass-remarks-output=%t -pass-remarks=inline \
; RUN:    -pass-remarks-missed=inline -pass-remarks-analysis=inline \
; RUN:    -annotate-inline-phase=false \
; RUN:    -pass-remarks-with-hotness 2>&1 | FileCheck %s
; RUN: cat %t | FileCheck -check-prefixes=YAML,YAML-NO-ANNOTATE %s

; RUN: opt < %s -S -passes=inliner-wrapper -pass-remarks-output=%t -pass-remarks=inline \
; RUN:    -pass-remarks-missed=inline -pass-remarks-analysis=inline \
; RUN:    -annotate-inline-phase \
; RUN:    -pass-remarks-with-hotness 2>&1 | FileCheck %s
; RUN: cat %t | FileCheck -check-prefixes=YAML,YAML-ANNOTATE %s

; Check the YAML file for inliner-generated passed and analysis remarks.  This
; is the input:

;  1     int foo() { return 1; }
;  2
;  3     int bar() {
;  4       return foo();
;  5     }

; CHECK: remark: /tmp/s.c:4:10: 'foo' inlined into 'bar' with (cost={{[0-9\-]+}}, threshold={{[0-9]+}}) at callsite bar:1:10; (hotness: 30)

; YAML:      --- !Passed
; YAML-NO-ANNOTATE-NEXT: Pass:            inline
; YAML-ANNOTATE-NEXT: Pass:            main-always-inline
; YAML-NEXT: Name:            Inlined
; YAML-NEXT: DebugLoc:        { File: '/tmp/s.c', Line: 4, Column: 10 }
; YAML-NEXT: Function:        bar
; YAML-NEXT: Hotness:         30
; YAML-NEXT: Args:
; YAML-NEXT:   - String: ''''
; YAML-NEXT:   - Callee: foo
; YAML-NEXT:     DebugLoc:        { File: '/tmp/s.c', Line: 1, Column: 0 }
; YAML-NEXT:   - String: ''' inlined into '''
; YAML-NEXT:   - Caller: bar
; YAML-NEXT:     DebugLoc:        { File: '/tmp/s.c', Line: 3, Column: 0 }
; YAML-NEXT:   - String: ''''
; YAML-NEXT:   - String: ' with '
; YAML-NEXT:   - String: '(cost='
; YAML-NEXT:   - Cost: '{{[0-9\-]+}}'
; YAML-NEXT:   - String: ', threshold='
; YAML-NEXT:   - Threshold: '{{[0-9]+}}'
; YAML-NEXT:   - String: ')'
; YAML-NEXT:   - String:          ' at callsite '
; YAML-NEXT:   - String:          bar
; YAML-NEXT:   - String:          ':'
; YAML-NEXT:   - Line:            '1'
; YAML-NEXT:   - String:          ':'
; YAML-NEXT:   - Column:          '10'
; YAML-NEXT:   - String:          ';'
; YAML-NEXT: ...

; ModuleID = '/tmp/s.c'
source_filename = "/tmp/s.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

; Function Attrs: nounwind ssp uwtable
define i32 @foo() #0 !dbg !7 {
entry:
  ret i32 1, !dbg !9
}

; Function Attrs: nounwind ssp uwtable
define i32 @bar() #0 !dbg !10 !prof !13 {
entry:
  %call = call i32 @foo(), !dbg !11
  ret i32 %call, !dbg !12
}

attributes #0 = { nounwind ssp uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="core2" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0.0 (trunk 282540) (llvm/trunk 282542)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2)
!1 = !DIFile(filename: "/tmp/s.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 4.0.0 (trunk 282540) (llvm/trunk 282542)"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 1, column: 13, scope: !7)
!10 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 3, type: !8, isLocal: false, isDefinition: true, scopeLine: 3, isOptimized: true, unit: !0, retainedNodes: !2)
!11 = !DILocation(line: 4, column: 10, scope: !10)
!12 = !DILocation(line: 4, column: 3, scope: !10)
!13 = !{!"function_entry_count", i64 30}
