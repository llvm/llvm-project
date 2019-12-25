; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   int foo(char a) { volatile short b = 0;  return b; }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c

; Function Attrs: nounwind
define dso_local i32 @foo(i8 signext) local_unnamed_addr #0 !dbg !7 {
  %2 = alloca i16, align 2
  call void @llvm.dbg.value(metadata i8 %0, metadata !13, metadata !DIExpression()), !dbg !17
  %3 = bitcast i16* %2 to i8*, !dbg !18
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %3), !dbg !18
  call void @llvm.dbg.declare(metadata i16* %2, metadata !14, metadata !DIExpression()), !dbg !19
  store volatile i16 0, i16* %2, align 2, !dbg !19, !tbaa !20
  %4 = load volatile i16, i16* %2, align 2, !dbg !24, !tbaa !20
  %5 = sext i16 %4 to i32, !dbg !24
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %3), !dbg !25
  ret i32 %5, !dbg !26
}

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   64
; CHECK-NEXT:        .long   64
; CHECK-NEXT:        .long   59
; CHECK-NEXT:        .long   1                       # BTF_KIND_INT(id = 1)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   16777224                # 0x1000008
; CHECK-NEXT:        .long   0                       # BTF_KIND_FUNC_PROTO(id = 2)
; CHECK-NEXT:        .long   218103809               # 0xd000001
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   6
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   8                       # BTF_KIND_INT(id = 3)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                # 0x1000020
; CHECK-NEXT:        .long   12                      # BTF_KIND_FUNC(id = 4)
; CHECK-NEXT:        .long   201326592               # 0xc000000
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .byte   0                       # string offset=0
; CHECK-NEXT:        .ascii  "char"                  # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .byte   97                      # string offset=6
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "int"                   # string offset=8
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "foo"                   # string offset=12
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  ".text"                 # string offset=16
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "/home/yhs/work/tests/llvm/bug/test.c" # string offset=22
; CHECK-NEXT:        .byte   0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #2

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { argmemonly nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 8.0.20181009 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/home/yhs/work/tests/llvm/bug")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.20181009 "}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!12 = !{!13, !14}
!13 = !DILocalVariable(name: "a", arg: 1, scope: !7, file: !1, line: 1, type: !11)
!14 = !DILocalVariable(name: "b", scope: !7, file: !1, line: 1, type: !15)
!15 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !16)
!16 = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
!17 = !DILocation(line: 1, column: 14, scope: !7)
!18 = !DILocation(line: 1, column: 19, scope: !7)
!19 = !DILocation(line: 1, column: 34, scope: !7)
!20 = !{!21, !21, i64 0}
!21 = !{!"short", !22, i64 0}
!22 = !{!"omnipotent char", !23, i64 0}
!23 = !{!"Simple C/C++ TBAA"}
!24 = !DILocation(line: 1, column: 49, scope: !7)
!25 = !DILocation(line: 1, column: 52, scope: !7)
!26 = !DILocation(line: 1, column: 42, scope: !7)
