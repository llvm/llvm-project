; RUN: llc -mtriple=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mtriple=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   void f(void) { }
; Compilation flag:
;   clang -target bpf -O2 -g -gdwarf-5 -gembed-source -S -emit-llvm t.c
;
; This test embeds the source code in the IR, so the line info should have
; correct reference to the lines in the string table.

; Function Attrs: norecurse nounwind readnone
define dso_local void @f() local_unnamed_addr #0 !dbg !7 {
entry:
  ret void, !dbg !10
}

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   35
; CHECK-NEXT:        .long   0                       # BTF_KIND_FUNC_PROTO(id = 1)
; CHECK-NEXT:        .long   218103808               # 0xd000000
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   1                       # BTF_KIND_FUNC(id = 2)
; CHECK-NEXT:        .long   201326593               # 0xc000001
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .byte   0                       # string offset=0
; CHECK-NEXT:        .byte   102                     # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  ".text"                 # string offset=3
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "/tmp/t.c"              # string offset=9
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "void f(void) { }"      # string offset=18
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .section        .BTF.ext,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   32
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   20
; CHECK-NEXT:        .long   20
; CHECK-NEXT:        .long   28
; CHECK-NEXT:        .long   48
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   8                       # FuncInfo
; CHECK-NEXT:        .long   3                       # FuncInfo section string offset=3
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   .Lfunc_begin0
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   16                      # LineInfo
; CHECK-NEXT:        .long   3                       # LineInfo section string offset=3
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   9
; CHECK-NEXT:        .long   18
; CHECK-NEXT:        .long   1040                    # Line 1 Col 16

attributes #0 = { norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 8.0.0 (trunk 344789) (llvm/trunk 344782)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "/tmp", checksumkind: CSK_MD5, checksum: "978599fafe3a080b456e3d95a3710359", source: "void f(void) { }\0A")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.0 (trunk 344789) (llvm/trunk 344782)"}
!7 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, isOptimized: true, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocation(line: 1, column: 16, scope: !7)
