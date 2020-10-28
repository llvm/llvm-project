; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK %s
; Source code:
;   typedef int __int;
;   typedef struct v3 { int a; __int b[4][4][4]; } __v3;
;   #define _(x) (__builtin_preserve_access_index(x))
;   int get_value(const int *arg);
;   int test(__v3 *arg) {
;     return get_value(_(&arg[1].b[2][3][2]));
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes test.c

target triple = "bpf"

%struct.v3 = type { i32, [4 x [4 x [4 x i32]]] }

; Function Attrs: nounwind
define dso_local i32 @test(%struct.v3* %arg) local_unnamed_addr #0 !dbg !23 {
entry:
  call void @llvm.dbg.value(metadata %struct.v3* %arg, metadata !27, metadata !DIExpression()), !dbg !28
  %0 = tail call %struct.v3* @llvm.preserve.array.access.index.p0s_struct.v3s.p0s_struct.v3s(%struct.v3* %arg, i32 0, i32 1), !dbg !29, !llvm.preserve.access.index !4
  %1 = tail call [4 x [4 x [4 x i32]]]* @llvm.preserve.struct.access.index.p0a4a4a4i32.p0s_struct.v3s(%struct.v3* %0, i32 1, i32 1), !dbg !29, !llvm.preserve.access.index !6
  %2 = tail call [4 x [4 x i32]]* @llvm.preserve.array.access.index.p0a4a4i32.p0a4a4a4i32([4 x [4 x [4 x i32]]]* %1, i32 1, i32 2), !dbg !29, !llvm.preserve.access.index !11
  %3 = tail call [4 x i32]* @llvm.preserve.array.access.index.p0a4i32.p0a4a4i32([4 x [4 x i32]]* %2, i32 1, i32 3), !dbg !29, !llvm.preserve.access.index !15
  %4 = tail call i32* @llvm.preserve.array.access.index.p0i32.p0a4i32([4 x i32]* %3, i32 1, i32 2), !dbg !29, !llvm.preserve.access.index !17
  %call = tail call i32 @get_value(i32* %4) #4, !dbg !30
  ret i32 %call, !dbg !31
}

; CHECK:             r2 = 448
; CHECK:             r1 += r2
; CHECK:             call get_value

; CHECK:             .long   6                       # BTF_KIND_STRUCT(id = [[TID1:[0-9]+]])

; CHECK:             .ascii  "v3"                    # string offset=6
; CHECK:             .ascii  ".text"                 # string offset=52
; CHECK:             .ascii  "1:1:2:3:2"             # string offset=58

; CHECK:             .long   16                      # FieldReloc
; CHECK-NEXT:        .long   52                      # Field reloc section string offset=52
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   [[TID1]]
; CHECK-NEXT:        .long   58
; CHECK-NEXT:        .long   0

declare dso_local i32 @get_value(i32*) local_unnamed_addr #1

; Function Attrs: nounwind readnone
declare %struct.v3* @llvm.preserve.array.access.index.p0s_struct.v3s.p0s_struct.v3s(%struct.v3*, i32, i32) #2

; Function Attrs: nounwind readnone
declare [4 x [4 x [4 x i32]]]* @llvm.preserve.struct.access.index.p0a4a4a4i32.p0s_struct.v3s(%struct.v3*, i32, i32) #2

; Function Attrs: nounwind readnone
declare [4 x [4 x i32]]* @llvm.preserve.array.access.index.p0a4a4i32.p0a4a4a4i32([4 x [4 x [4 x i32]]]*, i32, i32) #2

; Function Attrs: nounwind readnone
declare [4 x i32]* @llvm.preserve.array.access.index.p0a4i32.p0a4a4i32([4 x [4 x i32]]*, i32, i32) #2

; Function Attrs: nounwind readnone
declare i32* @llvm.preserve.array.access.index.p0i32.p0a4i32([4 x i32]*, i32, i32) #2

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind readnone speculatable willreturn }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!19, !20, !21}
!llvm.ident = !{!22}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (trunk 367256) (llvm/trunk 367266)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/llvm/cast")
!2 = !{}
!3 = !{!4, !11, !15, !17}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!5 = !DIDerivedType(tag: DW_TAG_typedef, name: "__v3", file: !1, line: 2, baseType: !6)
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "v3", file: !1, line: 2, size: 2080, elements: !7)
!7 = !{!8, !10}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !6, file: !1, line: 2, baseType: !9, size: 32)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !6, file: !1, line: 2, baseType: !11, size: 2048, offset: 32)
!11 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 2048, elements: !13)
!12 = !DIDerivedType(tag: DW_TAG_typedef, name: "__int", file: !1, line: 1, baseType: !9)
!13 = !{!14, !14, !14}
!14 = !DISubrange(count: 4)
!15 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 512, elements: !16)
!16 = !{!14, !14}
!17 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 128, elements: !18)
!18 = !{!14}
!19 = !{i32 2, !"Dwarf Version", i32 4}
!20 = !{i32 2, !"Debug Info Version", i32 3}
!21 = !{i32 1, !"wchar_size", i32 4}
!22 = !{!"clang version 10.0.0 (trunk 367256) (llvm/trunk 367266)"}
!23 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 5, type: !24, scopeLine: 5, flags: DIFlagPrototyped, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !26)
!24 = !DISubroutineType(types: !25)
!25 = !{!9, !4}
!26 = !{!27}
!27 = !DILocalVariable(name: "arg", arg: 1, scope: !23, file: !1, line: 5, type: !4)
!28 = !DILocation(line: 0, scope: !23)
!29 = !DILocation(line: 6, column: 20, scope: !23)
!30 = !DILocation(line: 6, column: 10, scope: !23)
!31 = !DILocation(line: 6, column: 3, scope: !23)
