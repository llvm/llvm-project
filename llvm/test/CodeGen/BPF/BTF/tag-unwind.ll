; RUN: llc -mtriple=bpfel -filetype=asm -o - %s | FileCheck %s

; A function that can unwind is tagged, so that the kernel knows that it says
; something in r3.
;
; Source:
;   void may_unwind(void);
;   int foo(void) { may_unwind(); return 0; }
;   int bar(void) __attribute__((nothrow)) { return 0; }
; Compilation flag:
;   clang -target bpf -O2 -g -fexceptions -funwind-tables -S -emit-llvm t.c

declare void @may_unwind()

define dso_local i32 @foo() uwtable !dbg !7 {
entry:
  call void @may_unwind(), !dbg !12
  ret i32 0, !dbg !12
}

define dso_local i32 @bar() nounwind uwtable !dbg !13 {
entry:
  ret i32 0, !dbg !14
}

; CHECK:        .long   {{[0-9]+}}                      # BTF_KIND_FUNC(id = [[FOO:[0-9]+]])
; CHECK:        .long   {{[0-9]+}}                      # BTF_KIND_DECL_TAG(id = {{[0-9]+}})
; CHECK-NEXT:   .long   285212672                       # 0x11000000
; CHECK-NEXT:   .long   [[FOO]]
; CHECK-NEXT:   .long   4294967295
; CHECK-NOT:    BTF_KIND_DECL_TAG
; CHECK:        .ascii  "unwind"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DILocation(line: 2, column: 18, scope: !7)
!13 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!14 = !DILocation(line: 3, column: 42, scope: !13)
