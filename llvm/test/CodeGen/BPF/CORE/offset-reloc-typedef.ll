; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck %s
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck %s
;
; Source code:
;   struct s { int a; int b; };
;   typedef struct s __s;
;   union u { __s c; __s d; };
;   typedef union u __u;
;   typedef __u arr_t[7];
;   typedef arr_t __arr;
;
;   #define _(x) (__builtin_preserve_access_index(x))
;   int get_value(const void *addr);
;   int test(__arr *arg) {
;     return get_value(_(&arg[1]->d.b));
;   }
; clang -target bpf -S -O2 -g -emit-llvm -Xclang -disable-llvm-passes test.c
; The offset reloc offset should be 12 from the base "arg".

target triple = "bpf"

%union.u = type { %struct.s }
%struct.s = type { i32, i32 }

; Function Attrs: nounwind
define dso_local i32 @test([7 x %union.u]* %arg) local_unnamed_addr #0 !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata [7 x %union.u]* %arg, metadata !28, metadata !DIExpression()), !dbg !29
  %0 = tail call [7 x %union.u]* @llvm.preserve.array.access.index.p0a7s_union.us.p0a7s_union.us([7 x %union.u]* elementtype([7 x %union.u]) %arg, i32 0, i32 1), !dbg !30, !llvm.preserve.access.index !14
  %arraydecay = getelementptr inbounds [7 x %union.u], [7 x %union.u]* %0, i64 0, i64 0, !dbg !30
  %1 = tail call %union.u* @llvm.preserve.union.access.index.p0s_union.us.p0s_union.us(%union.u* %arraydecay, i32 1), !dbg !30, !llvm.preserve.access.index !16
  %d = getelementptr inbounds %union.u, %union.u* %1, i64 0, i32 0, !dbg !30
  %2 = tail call i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.ss(%struct.s* elementtype(%struct.s) %d, i32 1, i32 1), !dbg !30, !llvm.preserve.access.index !20
  %3 = bitcast i32* %2 to i8*, !dbg !30
  %call = tail call i32 @get_value(i8* %3) #4, !dbg !31
  ret i32 %call, !dbg !32
}

; CHECK:         .cfi_startproc
; CHECK: [[RELOC:.Ltmp[0-9]+]]:
; CHECK:         r2 = 12
; CHECK:         r1 += r2
; CHECK:         call get_value

; CHECK:         .long   {{[0-9]+}}              # BTF_KIND_UNION(id = [[TYPE_ID:[0-9]+]])
; CHECK:         .ascii  ".text"                 # string offset=[[SEC_STR:[0-9]+]]
; CHECK-NEXT:    .byte   0
; CHECK:         .ascii  "1:1:1"                 # string offset=[[ACCESS_STR:[0-9]+]]
; CHECK-NEXT:    .byte   0
; CHECK:         .long   16                      # FieldReloc
; CHECK-NEXT:    .long   [[SEC_STR:[0-9]+]]      # Field reloc section string offset=[[SEC_STR:[0-9]+]]
; CHECK-NEXT:    .long   1
; CHECK-NEXT:    .long   [[RELOC:.Ltmp[0-9]+]]
; CHECK-NEXT:    .long   [[TYPE_ID:[0-9]+]]
; CHECK-NEXT:    .long   [[ACCESS_STR:[0-9]+]]
; CHECK-NEXT:    .long   0

declare dso_local i32 @get_value(i8*) local_unnamed_addr #1

; Function Attrs: nounwind readnone
declare [7 x %union.u]* @llvm.preserve.array.access.index.p0a7s_union.us.p0a7s_union.us([7 x %union.u]*, i32 immarg, i32 immarg) #2

; Function Attrs: nounwind readnone
declare %union.u* @llvm.preserve.union.access.index.p0s_union.us.p0s_union.us(%union.u*, i32 immarg) #2

; Function Attrs: nounwind readnone
declare i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.ss(%struct.s*, i32 immarg, i32 immarg) #2

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind readnone speculatable }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (trunk 366831) (llvm/trunk 366867)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/llvm/core-bugs")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 10.0.0 (trunk 366831) (llvm/trunk 366867)"}
!7 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 10, type: !8, scopeLine: 10, flags: DIFlagPrototyped, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !27)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DIDerivedType(tag: DW_TAG_typedef, name: "__arr", file: !1, line: 6, baseType: !13)
!13 = !DIDerivedType(tag: DW_TAG_typedef, name: "arr_t", file: !1, line: 5, baseType: !14)
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !15, size: 448, elements: !25)
!15 = !DIDerivedType(tag: DW_TAG_typedef, name: "__u", file: !1, line: 4, baseType: !16)
!16 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "u", file: !1, line: 3, size: 64, elements: !17)
!17 = !{!18, !24}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !16, file: !1, line: 3, baseType: !19, size: 64)
!19 = !DIDerivedType(tag: DW_TAG_typedef, name: "__s", file: !1, line: 2, baseType: !20)
!20 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s", file: !1, line: 1, size: 64, elements: !21)
!21 = !{!22, !23}
!22 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !20, file: !1, line: 1, baseType: !10, size: 32)
!23 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !20, file: !1, line: 1, baseType: !10, size: 32, offset: 32)
!24 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !16, file: !1, line: 3, baseType: !19, size: 64)
!25 = !{!26}
!26 = !DISubrange(count: 7)
!27 = !{!28}
!28 = !DILocalVariable(name: "arg", arg: 1, scope: !7, file: !1, line: 10, type: !11)
!29 = !DILocation(line: 0, scope: !7)
!30 = !DILocation(line: 11, column: 20, scope: !7)
!31 = !DILocation(line: 11, column: 10, scope: !7)
!32 = !DILocation(line: 11, column: 3, scope: !7)
