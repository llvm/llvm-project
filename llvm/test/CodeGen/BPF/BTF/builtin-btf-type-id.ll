; RUN: opt -O2 -mtriple=bpf-pc-linux %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK %s
;
; Source code:
;   static int (*bpf_log)(unsigned long tid, ptr data, int data_size) = (ptr)999;
;   struct {
;     char f1[100];
;     typeof(3) f2;
;   } tmp__abc = {1, 3};
;   void prog1() {
;     bpf_log(__builtin_btf_type_id(tmp__abc, 0), &tmp__abc, sizeof(tmp__abc));
;   }
;   void prog2() {
;     bpf_log(__builtin_btf_type_id(&tmp__abc, 0), &tmp__abc, sizeof(tmp__abc));
;   }
;   void prog3() {
;     bpf_log(__builtin_btf_type_id(tmp__abc.f1[3], 1), &tmp__abc, sizeof(tmp__abc));
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes test.c

@tmp__abc = dso_local global { <{ i8, i8, [98 x i8] }>, i32 } { <{ i8, i8, [98 x i8] }> <{ i8 1, i8 3, [98 x i8] zeroinitializer }>, i32 0 }, align 4, !dbg !0
@bpf_log = internal global ptr inttoptr (i64 999 to ptr), align 8, !dbg !17

; Function Attrs: nounwind
define dso_local void @prog1() #0 !dbg !28 {
entry:
  %0 = load ptr, ptr @bpf_log, align 8, !dbg !31, !tbaa !32
  %1 = call i64 @llvm.bpf.btf.type.id(i32 0, i64 0), !dbg !36, !llvm.preserve.access.index !7
  %call = call i32 %0(i64 %1, ptr getelementptr inbounds ({ <{ i8, i8, [98 x i8] }>, i32 }, ptr @tmp__abc, i32 0, i32 0, i32 0), i32 104), !dbg !31
  ret void, !dbg !37
}

; Function Attrs: nounwind readnone
declare i64 @llvm.bpf.btf.type.id(i32, i64) #1

; Function Attrs: nounwind
define dso_local void @prog2() #0 !dbg !38 {
entry:
  %0 = load ptr, ptr @bpf_log, align 8, !dbg !39, !tbaa !32
  %1 = call i64 @llvm.bpf.btf.type.id(i32 1, i64 0), !dbg !40, !llvm.preserve.access.index !6
  %call = call i32 %0(i64 %1, ptr getelementptr inbounds ({ <{ i8, i8, [98 x i8] }>, i32 }, ptr @tmp__abc, i32 0, i32 0, i32 0), i32 104), !dbg !39
  ret void, !dbg !41
}

; Function Attrs: nounwind
define dso_local void @prog3() #0 !dbg !42 {
entry:
  %0 = load ptr, ptr @bpf_log, align 8, !dbg !43, !tbaa !32
  %1 = call i64 @llvm.bpf.btf.type.id(i32 2, i64 1), !dbg !44, !llvm.preserve.access.index !11
  %call = call i32 %0(i64 %1, ptr getelementptr inbounds ({ <{ i8, i8, [98 x i8] }>, i32 }, ptr @tmp__abc, i32 0, i32 0, i32 0), i32 104), !dbg !43
  ret void, !dbg !45
}

; CHECK-LABEL:       prog1
; CHECK:             r1 = 3 ll
; CHECK-LABEL:       prog2
; CHECK:             r1 = 10 ll
; CHECK-LABEL:       prog3
; CHECK:             r1 = 4 ll

; CHECK:             .long   0                               # BTF_KIND_STRUCT(id = 3)
; CHECK-NEXT:        .long   67108866                        # 0x4000002
; CHECK-NEXT:        .long   104
; CHECK-NEXT:        .long   13
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   0                               # 0x0
; CHECK-NEXT:        .long   16
; CHECK-NEXT:        .long   7
; CHECK-NEXT:        .long   800                             # 0x320
; CHECK:             .long   19                              # BTF_KIND_INT(id = 4)
; CHECK:             .long   0                               # BTF_KIND_PTR(id = 10)
; CHECK-NEXT:        .long   33554432                        # 0x2000000
; CHECK-NEXT:        .long   3

; CHECK:             .ascii  ".text"                         # string offset=7
; CHECK:             .ascii  "f1"                            # string offset=13
; CHECK:             .ascii  "f2"                            # string offset=16
; CHECK:             .ascii  "char"                          # string offset=19
; CHECK:             .byte   48                              # string offset=48

; CHECK:             .long   16                              # FieldReloc
; CHECK-NEXT:        .long   7                               # Field reloc section string offset=7
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   48
; CHECK-NEXT:        .long   6
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   10
; CHECK-NEXT:        .long   48
; CHECK-NEXT:        .long   6
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   48
; CHECK-NEXT:        .long   7

attributes #0 = { nounwind "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!24, !25, !26}
!llvm.ident = !{!27}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "tmp__abc", scope: !2, file: !3, line: 5, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 12.0.0 (https://github.com/llvm/llvm-project.git 630c2da0e967e27e2a4c678dfc6e452a74141880)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !16, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/core")
!4 = !{}
!5 = !{!6, !11}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !3, line: 2, size: 832, elements: !8)
!8 = !{!9, !14}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "f1", scope: !7, file: !3, line: 3, baseType: !10, size: 800)
!10 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 800, elements: !12)
!11 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!12 = !{!13}
!13 = !DISubrange(count: 100)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "f2", scope: !7, file: !3, line: 4, baseType: !15, size: 32, offset: 800)
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !{!0, !17}
!17 = !DIGlobalVariableExpression(var: !18, expr: !DIExpression())
!18 = distinct !DIGlobalVariable(name: "bpf_log", scope: !2, file: !3, line: 1, type: !19, isLocal: true, isDefinition: true)
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20, size: 64)
!20 = !DISubroutineType(types: !21)
!21 = !{!15, !22, !23, !15}
!22 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!24 = !{i32 7, !"Dwarf Version", i32 4}
!25 = !{i32 2, !"Debug Info Version", i32 3}
!26 = !{i32 1, !"wchar_size", i32 4}
!27 = !{!"clang version 12.0.0 (https://github.com/llvm/llvm-project.git 630c2da0e967e27e2a4c678dfc6e452a74141880)"}
!28 = distinct !DISubprogram(name: "prog1", scope: !3, file: !3, line: 6, type: !29, scopeLine: 6, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!29 = !DISubroutineType(types: !30)
!30 = !{null}
!31 = !DILocation(line: 7, column: 3, scope: !28)
!32 = !{!33, !33, i64 0}
!33 = !{!"any pointer", !34, i64 0}
!34 = !{!"omnipotent char", !35, i64 0}
!35 = !{!"Simple C/C++ TBAA"}
!36 = !DILocation(line: 7, column: 11, scope: !28)
!37 = !DILocation(line: 8, column: 1, scope: !28)
!38 = distinct !DISubprogram(name: "prog2", scope: !3, file: !3, line: 9, type: !29, scopeLine: 9, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!39 = !DILocation(line: 10, column: 3, scope: !38)
!40 = !DILocation(line: 10, column: 11, scope: !38)
!41 = !DILocation(line: 11, column: 1, scope: !38)
!42 = distinct !DISubprogram(name: "prog3", scope: !3, file: !3, line: 12, type: !29, scopeLine: 12, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!43 = !DILocation(line: 13, column: 3, scope: !42)
!44 = !DILocation(line: 13, column: 11, scope: !42)
!45 = !DILocation(line: 14, column: 1, scope: !42)
