; RUN: opt %s -passes='sroa,verify' -S -o - | FileCheck %s
;
; Test that we can partial emit debug info for aggregates repeatedly
; split up by SROA.
;
;    // Compile with -O1
;    typedef struct {
;      int a;
;      long int b;
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
; CHECK: %[[I1:.*]] = alloca [12 x i8], align 4
; CHECK: call void @llvm.dbg.declare(metadata ptr %[[I1]], metadata ![[VAR:[0-9]+]], metadata !DIExpression(DW_OP_LLVM_fragment, 32, 96))
; CHECK: call void @llvm.dbg.value(metadata i32 %[[A:.*]], metadata ![[VAR]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32))
; CHECK: ret i32 %[[A]]
; Read Var and Piece:
; CHECK: ![[VAR]] = !DILocalVariable(name: "i1",{{.*}} line: 11,

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%struct.Outer = type { [2 x %struct.Inner] }
%struct.Inner = type { i32, i64 }

; Function Attrs: nounwind ssp uwtable
define i32 @foo(ptr byval(%struct.Outer) align 8 %outer) #0 !dbg !4 {
entry:
  %i1 = alloca %struct.Inner, align 8
  call void @llvm.dbg.declare(metadata ptr %outer, metadata !25, metadata !2), !dbg !26
  call void @llvm.dbg.declare(metadata ptr %i1, metadata !27, metadata !2), !dbg !28
  %arrayidx = getelementptr inbounds [2 x %struct.Inner], ptr %outer, i32 0, i64 1, !dbg !28
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %i1, ptr align 8 %arrayidx, i64 16, i1 false), !dbg !28
  %0 = load i32, ptr %i1, align 4, !dbg !29
  ret i32 %0, !dbg !29
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1) #2

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22, !23}
!llvm.ident = !{!24}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !{}, retainedTypes: !{}, globals: !{}, imports: !{})
!1 = !DIFile(filename: "sroasplit-1.c", directory: "")
!2 = !DIExpression()
!4 = distinct !DISubprogram(name: "foo", line: 10, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 10, file: !1, scope: !5, type: !6, retainedNodes: !{})
!5 = !DIFile(filename: "sroasplit-1.c", directory: "")
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_typedef, name: "Outer", line: 8, file: !1, baseType: !10)
!10 = !DICompositeType(tag: DW_TAG_structure_type, line: 6, size: 256, align: 64, file: !1, elements: !11)
!11 = !{!12}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "inner", line: 7, size: 256, align: 64, file: !1, scope: !10, baseType: !13)
!13 = !DICompositeType(tag: DW_TAG_array_type, size: 256, align: 64, baseType: !14, elements: !20)
!14 = !DIDerivedType(tag: DW_TAG_typedef, name: "Inner", line: 4, file: !1, baseType: !15)
!15 = !DICompositeType(tag: DW_TAG_structure_type, line: 1, size: 128, align: 64, file: !1, elements: !16)
!16 = !{!17, !18}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "a", line: 2, size: 32, align: 32, file: !1, scope: !15, baseType: !8)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "b", line: 3, size: 64, align: 64, offset: 64, file: !1, scope: !15, baseType: !19)
!19 = !DIBasicType(tag: DW_TAG_base_type, name: "long int", size: 64, align: 64, encoding: DW_ATE_signed)
!20 = !{!21}
!21 = !DISubrange(count: 2)
!22 = !{i32 2, !"Dwarf Version", i32 2}
!23 = !{i32 1, !"Debug Info Version", i32 3}
!24 = !{!"clang version 3.5.0 "}
!25 = !DILocalVariable(name: "outer", line: 10, arg: 1, scope: !4, file: !5, type: !9)
!26 = !DILocation(line: 10, scope: !4)
!27 = !DILocalVariable(name: "i1", line: 11, scope: !4, file: !5, type: !14)
!28 = !DILocation(line: 11, scope: !4)
!29 = !DILocation(line: 12, scope: !4)
