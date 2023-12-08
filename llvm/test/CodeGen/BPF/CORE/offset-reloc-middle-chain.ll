; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK %s
; Source code:
;   struct t1 {
;     int c;
;   };
;   struct s1 {
;     struct t1 b;
;   };
;   struct r1 {
;     struct s1 a;
;   };
;   #define _(x) __builtin_preserve_access_index(x)
;   void test1(ptr p1, ptr p2, ptr p3);
;   void test(struct r1 *arg) {
;     struct s1 *ps = _(&arg->a);
;     struct t1 *pt = _(&arg->a.b);
;     int *pi = _(&arg->a.b.c);
;     test1(ps, pt, pi);
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes test.c

target triple = "bpf"

%struct.r1 = type { %struct.s1 }
%struct.s1 = type { %struct.t1 }
%struct.t1 = type { i32 }

; Function Attrs: nounwind
define dso_local void @test(ptr %arg) local_unnamed_addr #0 !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata ptr %arg, metadata !22, metadata !DIExpression()), !dbg !29
  %0 = tail call ptr @llvm.preserve.struct.access.index.p0.s1s.p0.r1s(ptr elementtype(%struct.r1) %arg, i32 0, i32 0), !dbg !30, !llvm.preserve.access.index !11
  call void @llvm.dbg.value(metadata ptr %0, metadata !23, metadata !DIExpression()), !dbg !29
  %1 = tail call ptr @llvm.preserve.struct.access.index.p0.t1s.p0.s1s(ptr elementtype(%struct.s1) %0, i32 0, i32 0), !dbg !31, !llvm.preserve.access.index !14
  call void @llvm.dbg.value(metadata ptr %1, metadata !25, metadata !DIExpression()), !dbg !29
  %2 = tail call ptr @llvm.preserve.struct.access.index.p0.p0.t1s(ptr elementtype(%struct.t1) %1, i32 0, i32 0), !dbg !32, !llvm.preserve.access.index !17
  call void @llvm.dbg.value(metadata ptr %2, metadata !27, metadata !DIExpression()), !dbg !29
  tail call void @test1(ptr %0, ptr %1, ptr %2) #4, !dbg !36
  ret void, !dbg !37
}

; CHECK:             .long   1                       # BTF_KIND_STRUCT(id = 2)

; CHECK:             .ascii  "r1"                    # string offset=1
; CHECK:             .ascii  ".text"                 # string offset=29
; CHECK:             .ascii  "0:0"                   # string offset=72
; CHECK:             .ascii  "0:0:0"                 # string offset=76
; CHECK:             .ascii  "0:0:0:0"               # string offset=82

; CHECK:             .long   16                      # FieldReloc
; CHECK-NEXT:        .long   29                      # Field reloc section string offset=29
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   72
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   76
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   82
; CHECK-NEXT:        .long   0

; Function Attrs: nounwind readnone
declare ptr @llvm.preserve.struct.access.index.p0.s1s.p0.r1s(ptr, i32, i32) #1

; Function Attrs: nounwind readnone
declare ptr @llvm.preserve.struct.access.index.p0.t1s.p0.s1s(ptr, i32, i32) #1

; Function Attrs: nounwind readnone
declare ptr @llvm.preserve.struct.access.index.p0.p0.t1s(ptr, i32, i32) #1

declare dso_local void @test1(ptr, ptr, ptr) local_unnamed_addr #2

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readnone speculatable willreturn }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (https://github.com/llvm/llvm-project.git 42b3328a2368b38fba6bdb0c616fe6c5520e3bc5)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/core")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git 42b3328a2368b38fba6bdb0c616fe6c5520e3bc5)"}
!7 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 12, type: !8, scopeLine: 12, flags: DIFlagPrototyped, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !21)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "r1", file: !1, line: 7, size: 32, elements: !12)
!12 = !{!13}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !11, file: !1, line: 8, baseType: !14, size: 32)
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s1", file: !1, line: 4, size: 32, elements: !15)
!15 = !{!16}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !14, file: !1, line: 5, baseType: !17, size: 32)
!17 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t1", file: !1, line: 1, size: 32, elements: !18)
!18 = !{!19}
!19 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !17, file: !1, line: 2, baseType: !20, size: 32)
!20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!21 = !{!22, !23, !25, !27}
!22 = !DILocalVariable(name: "arg", arg: 1, scope: !7, file: !1, line: 12, type: !10)
!23 = !DILocalVariable(name: "ps", scope: !7, file: !1, line: 13, type: !24)
!24 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!25 = !DILocalVariable(name: "pt", scope: !7, file: !1, line: 14, type: !26)
!26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !17, size: 64)
!27 = !DILocalVariable(name: "pi", scope: !7, file: !1, line: 15, type: !28)
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20, size: 64)
!29 = !DILocation(line: 0, scope: !7)
!30 = !DILocation(line: 13, column: 19, scope: !7)
!31 = !DILocation(line: 14, column: 19, scope: !7)
!32 = !DILocation(line: 15, column: 13, scope: !7)
!33 = !DILocation(line: 16, column: 9, scope: !7)
!34 = !DILocation(line: 16, column: 13, scope: !7)
!35 = !DILocation(line: 16, column: 17, scope: !7)
!36 = !DILocation(line: 16, column: 3, scope: !7)
!37 = !DILocation(line: 17, column: 1, scope: !7)
