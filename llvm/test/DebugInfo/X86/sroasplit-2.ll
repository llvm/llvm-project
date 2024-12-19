; RUN: opt %s -passes='sroa,verify' -S -o - | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators %s -passes='sroa,verify' -S -o - | FileCheck %s
;
; Test that we can partial emit debug info for aggregates repeatedly
; split up by SROA.
;
;    // Compile with -O1
;    typedef struct {
;      int a;
;      int b;
;    } Inner;
;
;    typedef struct {
;      Inner inner[2];
;    } Outer;
;
;    int foo(Outer outer) {
;      Inner i1 = outer.inner[1];
;      return i1.a;
;    }
;

; Verify that SROA creates a variable piece when splitting i1.
; CHECK:  #dbg_value(i64 %outer.coerce0, ![[O:[0-9]+]], !DIExpression(DW_OP_LLVM_fragment, 0, 64),
; CHECK:  #dbg_value(i32 {{.*}}, ![[O]], !DIExpression(DW_OP_LLVM_fragment, 64, 32),
; CHECK:  #dbg_value(i32 {{.*}}, ![[O]], !DIExpression(DW_OP_LLVM_fragment, 96, 32),
; CHECK:  #dbg_value({{.*}}, ![[I1:[0-9]+]], !DIExpression(DW_OP_LLVM_fragment, 0, 32),
; CHECK-DAG: ![[O]] = !DILocalVariable(name: "outer",{{.*}} line: 10
; CHECK-DAG: ![[I1]] = !DILocalVariable(name: "i1",{{.*}} line: 11

; ModuleID = 'sroasplit-2.c'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%struct.Outer = type { [2 x %struct.Inner] }
%struct.Inner = type { i32, i32 }

; Function Attrs: nounwind ssp uwtable
define i32 @foo(i64 %outer.coerce0, i64 %outer.coerce1) #0 !dbg !4 {
  %outer = alloca %struct.Outer, align 8
  %i1 = alloca %struct.Inner, align 4
  %1 = getelementptr { i64, i64 }, ptr %outer, i32 0, i32 0
  store i64 %outer.coerce0, ptr %1
  %2 = getelementptr { i64, i64 }, ptr %outer, i32 0, i32 1
  store i64 %outer.coerce1, ptr %2
  call void @llvm.dbg.declare(metadata ptr %outer, metadata !24, metadata !2), !dbg !25
  call void @llvm.dbg.declare(metadata ptr %i1, metadata !26, metadata !2), !dbg !27
  %3 = getelementptr inbounds [2 x %struct.Inner], ptr %outer, i32 0, i64 1, !dbg !27
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %i1, ptr align 4 %3, i64 8, i1 false), !dbg !27
  %4 = load i32, ptr %i1, align 4, !dbg !28
  ret i32 %4, !dbg !28
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1) #2

attributes #0 = { nounwind ssp uwtable "frame-pointer"="all" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !22}
!llvm.ident = !{!23}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !{}, retainedTypes: !{}, globals: !{}, imports: !{})
!1 = !DIFile(filename: "sroasplit-2.c", directory: "")
!2 = !DIExpression()
!4 = distinct !DISubprogram(name: "foo", line: 10, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 10, file: !1, scope: !5, type: !6, retainedNodes: !{})
!5 = !DIFile(filename: "sroasplit-2.c", directory: "")
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_typedef, name: "Outer", line: 8, file: !1, baseType: !10)
!10 = !DICompositeType(tag: DW_TAG_structure_type, line: 6, size: 128, align: 32, file: !1, elements: !11)
!11 = !{!12}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "inner", line: 7, size: 128, align: 32, file: !1, scope: !10, baseType: !13)
!13 = !DICompositeType(tag: DW_TAG_array_type, size: 128, align: 32, baseType: !14, elements: !19)
!14 = !DIDerivedType(tag: DW_TAG_typedef, name: "Inner", line: 4, file: !1, baseType: !15)
!15 = !DICompositeType(tag: DW_TAG_structure_type, line: 1, size: 64, align: 32, file: !1, elements: !16)
!16 = !{!17, !18}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "a", line: 2, size: 32, align: 32, file: !1, scope: !15, baseType: !8)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "b", line: 3, size: 32, align: 32, offset: 32, file: !1, scope: !15, baseType: !8)
!19 = !{!20}
!20 = !DISubrange(count: 2)
!21 = !{i32 2, !"Dwarf Version", i32 2}
!22 = !{i32 1, !"Debug Info Version", i32 3}
!23 = !{!"clang version 3.5.0 "}
!24 = !DILocalVariable(name: "outer", line: 10, arg: 1, scope: !4, file: !5, type: !9)
!25 = !DILocation(line: 10, scope: !4)
!26 = !DILocalVariable(name: "i1", line: 11, scope: !4, file: !5, type: !14)
!27 = !DILocation(line: 11, scope: !4)
!28 = !DILocation(line: 12, scope: !4)
