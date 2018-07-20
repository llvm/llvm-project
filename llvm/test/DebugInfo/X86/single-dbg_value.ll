; RUN: %llc_dwarf -stop-after=livedebugvalues -o - %s \
; RUN:   | FileCheck %s --check-prefix=SANITY
; RUN: %llc_dwarf -march=x86-64 -o - %s -filetype=obj \
; RUN:   | llvm-dwarfdump -v -all - | FileCheck %s
;
; CHECK: .debug_info contents:
; CHECK: DW_TAG_variable
; CHECK-NEXT:   DW_AT_location [DW_FORM_data4] (
; CHECK-NEXT:     {{.*}}: DW_OP_reg0 RAX)
; CHECK-NEXT:   DW_AT_name{{.*}}"a"

; SANITY: DBG_VALUE
; SANITY-NOT: DBG_VALUE

; Compiled with -O:
;   void h(int);
;   int g();
;   void f() {
;     h(0);
;     int a = g();
;     h(a);
;   }

; ModuleID = 'test.ll'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

; Function Attrs: nounwind ssp uwtable
define void @f() #0 !dbg !4 {
entry:
  tail call void @h(i32 0) #2, !dbg !14
  %call = tail call i32 (...) @g() #2, !dbg !15
  tail call void @llvm.dbg.value(metadata i32 %call, metadata !8, metadata !16), !dbg !17
  tail call void @h(i32 %call) #2, !dbg !18
  ret void, !dbg !19
}

declare void @h(i32)

declare i32 @g(...)

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable "no-frame-pointer-elim"="true" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11, !12}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/Volumes/Data/llvm")
!2 = !{}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, isOptimized: true, unit: !0, retainedNodes: !7)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{!8}
!8 = !DILocalVariable(name: "a", scope: !4, file: !1, line: 5, type: !9)
!9 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !{i32 2, !"Dwarf Version", i32 2}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"PIC Level", i32 2}
!13 = !{!"clang version 3.9.0 "}
!14 = !DILocation(line: 4, column: 3, scope: !4)
!15 = !DILocation(line: 5, column: 11, scope: !4)
!16 = !DIExpression()
!17 = !DILocation(line: 5, column: 7, scope: !4)
!18 = !DILocation(line: 6, column: 3, scope: !4)
!19 = !DILocation(line: 7, column: 1, scope: !4)
