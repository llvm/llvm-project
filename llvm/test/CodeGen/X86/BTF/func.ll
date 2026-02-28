; RUN: llc -mtriple=x86_64-linux-gnu -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Verify BTF function types and .BTF.ext FuncInfo/LineInfo on x86_64.
;
; Source:
;   void f1(void) {}
; Compilation flag:
;   clang -target x86_64-linux-gnu -g -gbtf -S -emit-llvm t.c

define dso_local void @f1() !dbg !7 {
  ret void, !dbg !10
}

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                           # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   19
; CHECK-NEXT:        .long   0                               # BTF_KIND_FUNC_PROTO(id = 1)
; CHECK-NEXT:        .long   218103808                       # 0xd000000
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   1                               # BTF_KIND_FUNC(id = 2)
; CHECK-NEXT:        .long   201326593                       # 0xc000001
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .byte   0                               # string offset=0
; CHECK-NEXT:        .ascii  "f1"                            # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  ".text"                         # string offset=4
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "/tmp/t.c"                      # string offset=10
; CHECK-NEXT:        .byte   0
; CHECK:             .section        .BTF.ext,"",@progbits
; CHECK-NEXT:        .short  60319                           # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   32
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   20
; CHECK-NEXT:        .long   20
; CHECK-NEXT:        .long   28
; CHECK-NEXT:        .long   48
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   8                               # FuncInfo
; CHECK-NEXT:        .long   4                               # FuncInfo section string offset=4
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   .Lfunc_begin0
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   16                              # LineInfo
; CHECK-NEXT:        .long   4                               # LineInfo section string offset=4
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   10
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   1040                            # Line 1 Col 16

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 4, !"BTF", i32 1}
!7 = distinct !DISubprogram(name: "f1", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocation(line: 1, column: 16, scope: !7)
