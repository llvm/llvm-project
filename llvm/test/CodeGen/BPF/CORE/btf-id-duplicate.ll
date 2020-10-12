; RUN: opt -O2 -mtriple=bpf-pc-linux %s | llvm-dis > %t1
; RUN: llc -o - %t1 | FileCheck %s
; RUN: opt -passes='default<O2>' -mtriple=bpf-pc-linux %s | llvm-dis > %t1
; RUN: llc -o - %t1 | FileCheck %s
;
; Source:
;   struct s1 { int a; int b; };
;   int foo(struct s1 *arg) { return __builtin_btf_type_id(*arg, 0); }
;   int bar(struct s1 *arg) { return __builtin_btf_type_id(*arg, 0); }
; Compilation flag:
;   clang -target bpf -O2 -S -emit-llvm -g -Xclang -disable-llvm-passes test.c

%struct.s1 = type { i32, i32 }

; Function Attrs: nounwind
define dso_local i32 @foo(%struct.s1* %arg) #0 !dbg !7 {
entry:
  %arg.addr = alloca %struct.s1*, align 8
  store %struct.s1* %arg, %struct.s1** %arg.addr, align 8, !tbaa !18
  call void @llvm.dbg.declare(metadata %struct.s1** %arg.addr, metadata !17, metadata !DIExpression()), !dbg !22
  %0 = call i32 @llvm.bpf.btf.type.id(i32 0, i64 0), !dbg !23, !llvm.preserve.access.index !12
  ret i32 %0, !dbg !24
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.bpf.btf.type.id(i32, i64) #2

; Function Attrs: nounwind
define dso_local i32 @bar(%struct.s1* %arg) #0 !dbg !25 {
entry:
  %arg.addr = alloca %struct.s1*, align 8
  store %struct.s1* %arg, %struct.s1** %arg.addr, align 8, !tbaa !18
  call void @llvm.dbg.declare(metadata %struct.s1** %arg.addr, metadata !27, metadata !DIExpression()), !dbg !28
  %0 = call i32 @llvm.bpf.btf.type.id(i32 1, i64 0), !dbg !29, !llvm.preserve.access.index !12
  ret i32 %0, !dbg !30
}

; CHECK:             .long   1                               # BTF_KIND_STRUCT(id = 2)

; CHECK:             .ascii  "s1"                            # string offset=1
; CHECK:             .ascii  ".text"                         # string offset=20
; CHECK:             .byte   48                              # string offset=26

; CHECK:             .long   16                              # FieldReloc
; CHECK-NEXT:        .long   20                              # Field reloc section string offset=20
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   26
; CHECK-NEXT:        .long   6
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   26
; CHECK-NEXT:        .long   6

attributes #0 = { nounwind "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0 (https://github.com/llvm/llvm-project.git 80a3f7beebd8caab358ff063526ae2d26467c029)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/home/yhs/work/tests/dup")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0 (https://github.com/llvm/llvm-project.git 80a3f7beebd8caab358ff063526ae2d26467c029)"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !16)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s1", file: !1, line: 1, size: 64, elements: !13)
!13 = !{!14, !15}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !12, file: !1, line: 1, baseType: !10, size: 32)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !12, file: !1, line: 1, baseType: !10, size: 32, offset: 32)
!16 = !{!17}
!17 = !DILocalVariable(name: "arg", arg: 1, scope: !7, file: !1, line: 2, type: !11)
!18 = !{!19, !19, i64 0}
!19 = !{!"any pointer", !20, i64 0}
!20 = !{!"omnipotent char", !21, i64 0}
!21 = !{!"Simple C/C++ TBAA"}
!22 = !DILocation(line: 2, column: 20, scope: !7)
!23 = !DILocation(line: 2, column: 34, scope: !7)
!24 = !DILocation(line: 2, column: 27, scope: !7)
!25 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !26)
!26 = !{!27}
!27 = !DILocalVariable(name: "arg", arg: 1, scope: !25, file: !1, line: 3, type: !11)
!28 = !DILocation(line: 3, column: 20, scope: !25)
!29 = !DILocation(line: 3, column: 34, scope: !25)
!30 = !DILocation(line: 3, column: 27, scope: !25)
