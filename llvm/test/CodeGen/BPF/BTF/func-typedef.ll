; RUN: llc -mtriple=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mtriple=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   typedef int _int;
;   typedef _int __int;
;   __int f(__int a) { return a; }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm t.c

; Function Attrs: nounwind readnone
define dso_local i32 @f(i32 returned %a) local_unnamed_addr #0 !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32 %a, metadata !14, metadata !DIExpression()), !dbg !15
  ret i32 %a, !dbg !16
}

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   72
; CHECK-NEXT:        .long   72
; CHECK-NEXT:        .long   35
; CHECK-NEXT:        .long   1                       # BTF_KIND_TYPEDEF(id = 1)
; CHECK-NEXT:        .long   134217728               # 0x8000000
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   7                       # BTF_KIND_TYPEDEF(id = 2)
; CHECK-NEXT:        .long   134217728               # 0x8000000
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   12                      # BTF_KIND_INT(id = 3)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                # 0x1000020
; CHECK-NEXT:        .long   0                       # BTF_KIND_FUNC_PROTO(id = 4)
; CHECK-NEXT:        .long   218103809               # 0xd000001
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   16
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   18                      # BTF_KIND_FUNC(id = 5)
; CHECK-NEXT:        .long   201326593               # 0xc000001
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .byte   0                       # string offset=0
; CHECK-NEXT:        .ascii  "__int"                 # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "_int"                  # string offset=7
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "int"                   # string offset=12
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .byte   97                      # string offset=16
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .byte   102                     # string offset=18
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  ".text"                 # string offset=20
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "/tmp/t.c"              # string offset=26
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .section        .BTF.ext,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   32
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   20
; CHECK-NEXT:        .long   20
; CHECK-NEXT:        .long   44
; CHECK-NEXT:        .long   64
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   8                       # FuncInfo
; CHECK-NEXT:        .long   20                      # FuncInfo section string offset=20
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   .Lfunc_begin0
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   16                      # LineInfo
; CHECK-NEXT:        .long   20                      # LineInfo section string offset=20
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   .Lfunc_begin0
; CHECK-NEXT:        .long   26
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   3072                    # Line 3 Col 0
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   26
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   3092                    # Line 3 Col 20


; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 8.0.0 (trunk 344789) (llvm/trunk 344782)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.0 (trunk 344789) (llvm/trunk 344782)"}
!7 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 3, type: !8, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !13)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIDerivedType(tag: DW_TAG_typedef, name: "__int", file: !1, line: 2, baseType: !11)
!11 = !DIDerivedType(tag: DW_TAG_typedef, name: "_int", file: !1, line: 1, baseType: !12)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{!14}
!14 = !DILocalVariable(name: "a", arg: 1, scope: !7, file: !1, line: 3, type: !10)
!15 = !DILocation(line: 3, column: 15, scope: !7)
!16 = !DILocation(line: 3, column: 20, scope: !7)
