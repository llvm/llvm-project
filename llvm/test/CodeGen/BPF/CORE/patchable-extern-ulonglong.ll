; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; Source code:
;   extern __attribute__((section(".BPF.patchable_externs"))) unsigned long long a;
;   int foo() { return a; }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c

@a = external dso_local local_unnamed_addr global i64, section ".BPF.patchable_externs", align 8

; Function Attrs: norecurse nounwind readonly
define dso_local i32 @foo() local_unnamed_addr #0 !dbg !7 {
  %1 = load i64, i64* @a, align 8, !dbg !11, !tbaa !12
  %2 = trunc i64 %1 to i32, !dbg !11
; CHECK:             r0 = 0 ll
; CHECK-NEXT:        exit
  ret i32 %2, !dbg !16
}

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   40
; CHECK-NEXT:        .long   40
; CHECK-NEXT:        .long   54
; CHECK-NEXT:        .long   0                       # BTF_KIND_FUNC_PROTO(id = 1)
; CHECK-NEXT:        .long   218103808               # 0xd000000
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   1                       # BTF_KIND_INT(id = 2)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                # 0x1000020
; CHECK-NEXT:        .long   5                       # BTF_KIND_FUNC(id = 3)
; CHECK-NEXT:        .long   201326592               # 0xc000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .byte   0                       # string offset=0
; CHECK-NEXT:        .ascii  "int"                   # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "foo"                   # string offset=5
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  ".text"                 # string offset=9
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .byte   97                      # string offset=15
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "/tmp/home/yhs/work/tests/llvm/test.c" # string offset=17
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .section        .BTF.ext,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   40
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   20
; CHECK-NEXT:        .long   20
; CHECK-NEXT:        .long   28
; CHECK-NEXT:        .long   48
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   48
; CHECK-NEXT:        .long   20
; CHECK-NEXT:        .long   8                       # FuncInfo
; CHECK-NEXT:        .long   9                       # FuncInfo section string offset=9
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   .Lfunc_begin0
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   16                      # LineInfo
; CHECK-NEXT:        .long   9                       # LineInfo section string offset=9
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   17
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   2061                    # Line 2 Col 13
; CHECK-NEXT:        .long   8                       # ExternReloc
; CHECK-NEXT:        .long   9                       # Extern reloc section string offset=9
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   15

attributes #0 = { norecurse nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 8.0.20181009 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/llvm")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.20181009 "}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !8, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: true, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 2, column: 20, scope: !7)
!12 = !{!13, !13, i64 0}
!13 = !{!"long long", !14, i64 0}
!14 = !{!"omnipotent char", !15, i64 0}
!15 = !{!"Simple C/C++ TBAA"}
!16 = !DILocation(line: 2, column: 13, scope: !7)
