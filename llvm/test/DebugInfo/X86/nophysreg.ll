; RUN: llc -filetype=obj < %s | llvm-dwarfdump -debug-info - | FileCheck %s
;
; PR22296: In this testcase the DBG_VALUE describing "p5" becomes unavailable
; because the register its address is in is clobbered and we (currently) aren't
; smart enough to realize that the value is rematerialized immediately after the
; DBG_VALUE and/or is actually a stack slot.
;
; Test that we handle this situation gracefully by omitting the DW_AT_location
; and not asserting.
; Note that this check may XPASS in the future if DbgValueHistoryCalculator
; becoms smarter. That would be fine, too.
;
; CHECK: DW_TAG_subprogram
; CHECK: linkage_name{{.*}}_Z2f21A
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_AT_location
; CHECK-NEXT: DW_AT_name {{.*}}"p5"
;
; // Compile at -O1
; struct A {
;   int *m1;
;   int m2;
; };
;
; void f1(int *p1, int p2);
; void __attribute__((always_inline)) f2(A p5) { f1(p5.m1, p5.m2); }
;
; void func(void*);
; void func(const int &, const int&);
; int cond();
; void f() {
;   while (cond()) {
;     int x;
;     func(x, 0);
;     while (cond()) {
;       char y;
;       func(&y);
;       char j;
;       func(&j);
;       char I;
;       func(&I);
;       func(0, 0);
;       A g;
;       g.m1 = &x;
;       f2(g);
;     }
;   }
; }
; ModuleID = 'test.cpp'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

%struct.A = type { i32*, i32 }

; Function Attrs: alwaysinline ssp uwtable
define void @_Z2f21A(i32* %p5.coerce0, i32 %p5.coerce1) #0 !dbg !11 {
entry:
  tail call void @llvm.dbg.value(metadata i32* %p5.coerce0, metadata !16, metadata !33), !dbg !34
  tail call void @llvm.dbg.value(metadata i32 %p5.coerce1, metadata !16, metadata !35), !dbg !34
  tail call void @llvm.dbg.declare(metadata %struct.A* undef, metadata !16, metadata !36), !dbg !34
  tail call void @_Z2f1Pii(i32* %p5.coerce0, i32 %p5.coerce1), !dbg !37
  ret void, !dbg !38
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @_Z2f1Pii(i32*, i32) #2

; Function Attrs: ssp uwtable
define void @_Z1fv() #3 !dbg !17 {
entry:
  %x = alloca i32, align 4
  %ref.tmp = alloca i32, align 4
  %y = alloca i8, align 1
  %j = alloca i8, align 1
  %I = alloca i8, align 1
  %ref.tmp5 = alloca i32, align 4
  %ref.tmp6 = alloca i32, align 4
  %call11 = call i32 @_Z4condv(), !dbg !39
  %tobool12 = icmp eq i32 %call11, 0, !dbg !39
  br i1 %tobool12, label %while.end7, label %while.body, !dbg !40

while.cond.loopexit:                              ; preds = %while.body4, %while.body
  %call = call i32 @_Z4condv(), !dbg !39
  %tobool = icmp eq i32 %call, 0, !dbg !39
  br i1 %tobool, label %while.end7, label %while.body, !dbg !40

while.body:                                       ; preds = %entry, %while.cond.loopexit
  store i32 0, i32* %ref.tmp, align 4, !dbg !41, !tbaa !42
  call void @llvm.dbg.value(metadata i32* %x, metadata !21, metadata !DIExpression(DW_OP_deref)), !dbg !46
  call void @_Z4funcRKiS0_(i32* dereferenceable(4) %x, i32* dereferenceable(4) %ref.tmp), !dbg !47
  %call29 = call i32 @_Z4condv(), !dbg !48
  %tobool310 = icmp eq i32 %call29, 0, !dbg !48
  br i1 %tobool310, label %while.cond.loopexit, label %while.body4, !dbg !49

while.body4:                                      ; preds = %while.body, %while.body4
  call void @llvm.dbg.value(metadata i8* %y, metadata !23, metadata !DIExpression(DW_OP_deref)), !dbg !50
  call void @_Z4funcPv(i8* %y), !dbg !51
  call void @llvm.dbg.value(metadata i8* %j, metadata !26, metadata !DIExpression(DW_OP_deref)), !dbg !52
  call void @_Z4funcPv(i8* %j), !dbg !53
  call void @llvm.dbg.value(metadata i8* %I, metadata !27, metadata !DIExpression(DW_OP_deref)), !dbg !54
  call void @_Z4funcPv(i8* %I), !dbg !55
  store i32 0, i32* %ref.tmp5, align 4, !dbg !56, !tbaa !42
  store i32 0, i32* %ref.tmp6, align 4, !dbg !57, !tbaa !42
  call void @_Z4funcRKiS0_(i32* dereferenceable(4) %ref.tmp5, i32* dereferenceable(4) %ref.tmp6), !dbg !58
  call void @llvm.dbg.declare(metadata %struct.A* undef, metadata !28, metadata !36), !dbg !59
  call void @llvm.dbg.value(metadata i32* %x, metadata !28, metadata !33), !dbg !59
  call void @llvm.dbg.value(metadata i32* %x, metadata !21, metadata !DIExpression(DW_OP_deref)), !dbg !46
  call void @llvm.dbg.value(metadata i32* %x, metadata !60, metadata !33), !dbg !62
  call void @llvm.dbg.value(metadata i32 undef, metadata !60, metadata !35), !dbg !62
  call void @llvm.dbg.declare(metadata %struct.A* undef, metadata !60, metadata !36), !dbg !62
  call void @_Z2f1Pii(i32* %x, i32 undef), !dbg !63
  %call2 = call i32 @_Z4condv(), !dbg !48
  %tobool3 = icmp eq i32 %call2, 0, !dbg !48
  br i1 %tobool3, label %while.cond.loopexit, label %while.body4, !dbg !49

while.end7:                                       ; preds = %while.cond.loopexit, %entry
  ret void, !dbg !64
}

declare i32 @_Z4condv()

declare void @_Z4funcRKiS0_(i32* dereferenceable(4), i32* dereferenceable(4))

declare void @_Z4funcPv(i8*)

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { alwaysinline ssp uwtable }
attributes #1 = { nounwind readnone }
attributes #3 = { ssp uwtable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!29, !30, !31}
!llvm.ident = !{!32}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.7.0 (trunk 227088) (llvm/trunk 227091)", isOptimized: true, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "test.cpp", directory: "")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "A", line: 1, size: 128, align: 64, file: !1, elements: !5, identifier: "_ZTS1A")
!5 = !{!6, !9}
!6 = !DIDerivedType(tag: DW_TAG_member, name: "m1", line: 2, size: 64, align: 64, file: !1, scope: !4, baseType: !7)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !8)
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "m2", line: 3, size: 32, align: 32, offset: 64, file: !1, scope: !4, baseType: !8)
!11 = distinct !DISubprogram(name: "f2", linkageName: "_Z2f21A", line: 7, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 7, file: !1, scope: !12, type: !13, variables: !15)
!12 = !DIFile(filename: "test.cpp", directory: "")
!13 = !DISubroutineType(types: !14)
!14 = !{null, !4}
!15 = !{!16}
!16 = !DILocalVariable(name: "p5", line: 7, arg: 1, scope: !11, file: !12, type: !4)
!17 = distinct !DISubprogram(name: "f", linkageName: "_Z1fv", line: 12, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 12, file: !1, scope: !12, type: !18, variables: !20)
!18 = !DISubroutineType(types: !19)
!19 = !{null}
!20 = !{!21, !23, !26, !27, !28}
!21 = !DILocalVariable(name: "x", line: 14, scope: !22, file: !12, type: !8)
!22 = distinct !DILexicalBlock(line: 13, column: 18, file: !1, scope: !17)
!23 = !DILocalVariable(name: "y", line: 17, scope: !24, file: !12, type: !25)
!24 = distinct !DILexicalBlock(line: 16, column: 20, file: !1, scope: !22)
!25 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!26 = !DILocalVariable(name: "j", line: 19, scope: !24, file: !12, type: !25)
!27 = !DILocalVariable(name: "I", line: 21, scope: !24, file: !12, type: !25)
!28 = !DILocalVariable(name: "g", line: 24, scope: !24, file: !12, type: !4)
!29 = !{i32 2, !"Dwarf Version", i32 2}
!30 = !{i32 2, !"Debug Info Version", i32 3}
!31 = !{i32 1, !"PIC Level", i32 2}
!32 = !{!"clang version 3.7.0 (trunk 227088) (llvm/trunk 227091)"}
!33 = !DIExpression(DW_OP_LLVM_fragment, 0, 8)
!34 = !DILocation(line: 7, column: 42, scope: !11)
!35 = !DIExpression(DW_OP_LLVM_fragment, 8, 4)
!36 = !DIExpression()
!37 = !DILocation(line: 7, column: 48, scope: !11)
!38 = !DILocation(line: 7, column: 66, scope: !11)
!39 = !DILocation(line: 13, column: 10, scope: !17)
!40 = !DILocation(line: 13, column: 3, scope: !17)
!41 = !DILocation(line: 15, column: 13, scope: !22)
!42 = !{!43, !43, i64 0}
!43 = !{!"int", !44, i64 0}
!44 = !{!"omnipotent char", !45, i64 0}
!45 = !{!"Simple C/C++ TBAA"}
!46 = !DILocation(line: 14, column: 9, scope: !22)
!47 = !DILocation(line: 15, column: 5, scope: !22)
!48 = !DILocation(line: 16, column: 12, scope: !22)
!49 = !DILocation(line: 16, column: 5, scope: !22)
!50 = !DILocation(line: 17, column: 12, scope: !24)
!51 = !DILocation(line: 18, column: 7, scope: !24)
!52 = !DILocation(line: 19, column: 12, scope: !24)
!53 = !DILocation(line: 20, column: 7, scope: !24)
!54 = !DILocation(line: 21, column: 12, scope: !24)
!55 = !DILocation(line: 22, column: 7, scope: !24)
!56 = !DILocation(line: 23, column: 12, scope: !24)
!57 = !DILocation(line: 23, column: 15, scope: !24)
!58 = !DILocation(line: 23, column: 7, scope: !24)
!59 = !DILocation(line: 24, column: 9, scope: !24)
!60 = !DILocalVariable(name: "p5", line: 7, arg: 1, scope: !11, file: !12, type: !4)
!61 = distinct !DILocation(line: 26, column: 7, scope: !24)
!62 = !DILocation(line: 7, column: 42, scope: !11, inlinedAt: !61)
!63 = !DILocation(line: 7, column: 48, scope: !11, inlinedAt: !61)
!64 = !DILocation(line: 29, column: 1, scope: !17)
