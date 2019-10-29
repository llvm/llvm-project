; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfel -mattr=+alu32 -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -mattr=+alu32 -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; Source code:
;   struct v1 {int a; int b;};
;   typedef struct v1 __v1;
;   typedef __v1 arr[4][4];
;   struct v3 { char c; int d[100]; };
;   #define _(x) (__builtin_preserve_access_index(x))
;   #define cast_to_arr(x) ((arr *)(x))
;   int get_value(const int *arg);
;   int test(struct v3 *arg) {
;     return get_value(_(&cast_to_arr(&arg->d[0])[0][2][3].b));
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c

%struct.v3 = type { i8, [100 x i32] }
%struct.v1 = type { i32, i32 }

; Function Attrs: nounwind
define dso_local i32 @test(%struct.v3* %arg) local_unnamed_addr #0 !dbg !24 {
entry:
  call void @llvm.dbg.value(metadata %struct.v3* %arg, metadata !34, metadata !DIExpression()), !dbg !35
  %0 = tail call [100 x i32]* @llvm.preserve.struct.access.index.p0a100i32.p0s_struct.v3s(%struct.v3* %arg, i32 1, i32 1), !dbg !36, !llvm.preserve.access.index !28
  %1 = tail call i32* @llvm.preserve.array.access.index.p0i32.p0a100i32([100 x i32]* %0, i32 1, i32 0), !dbg !36, !llvm.preserve.access.index !15
  %2 = bitcast i32* %1 to [4 x [4 x %struct.v1]]*, !dbg !36
  %3 = tail call [4 x [4 x %struct.v1]]* @llvm.preserve.array.access.index.p0a4a4s_struct.v1s.p0a4a4s_struct.v1s([4 x [4 x %struct.v1]]* %2, i32 0, i32 0), !dbg !36, !llvm.preserve.access.index !4
  %4 = tail call [4 x %struct.v1]* @llvm.preserve.array.access.index.p0a4s_struct.v1s.p0a4a4s_struct.v1s([4 x [4 x %struct.v1]]* %3, i32 1, i32 2), !dbg !36, !llvm.preserve.access.index !5
  %5 = tail call %struct.v1* @llvm.preserve.array.access.index.p0s_struct.v1s.p0a4s_struct.v1s([4 x %struct.v1]* %4, i32 1, i32 3), !dbg !36, !llvm.preserve.access.index !18
  %6 = tail call i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.v1s(%struct.v1* %5, i32 1, i32 1), !dbg !36, !llvm.preserve.access.index !8
  %call = tail call i32 @get_value(i32* %6) #4, !dbg !37
  ret i32 %call, !dbg !38
}

; CHECK:              r2 = 4
; CHECK:              r1 += r2
; CHECK:              r2 = 92
; CHECK:              r1 += r2
; CHECK:              call get_value

; CHECK:              .long   1                       # BTF_KIND_STRUCT(id = [[TID1:[0-9]+]])
; CHECK:              .long   100                     # BTF_KIND_STRUCT(id = [[TID2:[0-9]+]])

; CHECK:              .ascii  "v3"                    # string offset=1
; CHECK:              .ascii  ".text"                 # string offset=46
; CHECK:              .ascii  "0:1:0"                 # string offset=52
; CHECK:              .ascii  "v1"                    # string offset=100
; CHECK:              .ascii  "11:1"                  # string offset=107

; CHECK:              .long   16                      # FieldReloc
; CHECK-NEXT:         .long   46                      # Field reloc section string offset=46
; CHECK-NEXT:         .long   2
; CHECK-NEXT:         .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:         .long   [[TID1]]
; CHECK-NEXT:         .long   52
; CHECK-NEXT:         .long   0
; CHECK-NEXT:         .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:         .long   [[TID2]]
; CHECK-NEXT:         .long   107
; CHECK-NEXT:         .long   0

declare dso_local i32 @get_value(i32*) local_unnamed_addr #1

; Function Attrs: nounwind readnone
declare [100 x i32]* @llvm.preserve.struct.access.index.p0a100i32.p0s_struct.v3s(%struct.v3*, i32, i32) #2

; Function Attrs: nounwind readnone
declare i32* @llvm.preserve.array.access.index.p0i32.p0a100i32([100 x i32]*, i32, i32) #2

; Function Attrs: nounwind readnone
declare [4 x [4 x %struct.v1]]* @llvm.preserve.array.access.index.p0a4a4s_struct.v1s.p0a4a4s_struct.v1s([4 x [4 x %struct.v1]]*, i32, i32) #2

; Function Attrs: nounwind readnone
declare [4 x %struct.v1]* @llvm.preserve.array.access.index.p0a4s_struct.v1s.p0a4a4s_struct.v1s([4 x [4 x %struct.v1]]*, i32, i32) #2

; Function Attrs: nounwind readnone
declare %struct.v1* @llvm.preserve.array.access.index.p0s_struct.v1s.p0a4s_struct.v1s([4 x %struct.v1]*, i32, i32) #2

; Function Attrs: nounwind readnone
declare i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.v1s(%struct.v1*, i32, i32) #2

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind readnone speculatable willreturn }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!20, !21, !22}
!llvm.ident = !{!23}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (trunk 367256) (llvm/trunk 367266)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/llvm/cast")
!2 = !{}
!3 = !{!4, !15, !5, !18}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!5 = !DIDerivedType(tag: DW_TAG_typedef, name: "arr", file: !1, line: 3, baseType: !6)
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 1024, elements: !13)
!7 = !DIDerivedType(tag: DW_TAG_typedef, name: "__v1", file: !1, line: 2, baseType: !8)
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "v1", file: !1, line: 1, size: 64, elements: !9)
!9 = !{!10, !12}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !8, file: !1, line: 1, baseType: !11, size: 32)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !8, file: !1, line: 1, baseType: !11, size: 32, offset: 32)
!13 = !{!14, !14}
!14 = !DISubrange(count: 4)
!15 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 3200, elements: !16)
!16 = !{!17}
!17 = !DISubrange(count: 100)
!18 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 256, elements: !19)
!19 = !{!14}
!20 = !{i32 2, !"Dwarf Version", i32 4}
!21 = !{i32 2, !"Debug Info Version", i32 3}
!22 = !{i32 1, !"wchar_size", i32 4}
!23 = !{!"clang version 10.0.0 (trunk 367256) (llvm/trunk 367266)"}
!24 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 8, type: !25, scopeLine: 8, flags: DIFlagPrototyped, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !33)
!25 = !DISubroutineType(types: !26)
!26 = !{!11, !27}
!27 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !28, size: 64)
!28 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "v3", file: !1, line: 4, size: 3232, elements: !29)
!29 = !{!30, !32}
!30 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !28, file: !1, line: 4, baseType: !31, size: 8)
!31 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!32 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !28, file: !1, line: 4, baseType: !15, size: 3200, offset: 32)
!33 = !{!34}
!34 = !DILocalVariable(name: "arg", arg: 1, scope: !24, file: !1, line: 8, type: !27)
!35 = !DILocation(line: 0, scope: !24)
!36 = !DILocation(line: 9, column: 20, scope: !24)
!37 = !DILocation(line: 9, column: 10, scope: !24)
!38 = !DILocation(line: 9, column: 3, scope: !24)
