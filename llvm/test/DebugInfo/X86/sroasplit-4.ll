; RUN: opt -sroa < %s -S -o - | FileCheck %s
;
; Test that recursively splitting an alloca updates the debug info correctly.
; CHECK: %[[T:.*]] = load i64, i64* @t, align 8
; CHECK: call void @llvm.dbg.value(metadata i64 %[[T]], i64 0, metadata ![[Y:.*]], metadata ![[P1:.*]])
; CHECK: %[[T1:.*]] = load i64, i64* @t, align 8
; CHECK: call void @llvm.dbg.value(metadata i64 %[[T1]], i64 0, metadata ![[Y]], metadata ![[P2:.*]])
; CHECK: call void @llvm.dbg.value(metadata i64 %[[T]], i64 0, metadata ![[R:.*]], metadata ![[P3:.*]])
; CHECK: call void @llvm.dbg.value(metadata i64 %[[T1]], i64 0, metadata ![[R]], metadata ![[P4:.*]])
; CHECK: ![[P1]] = !DIExpression(DW_OP_bit_piece, 0, 64)
; CHECK: ![[P2]] = !DIExpression(DW_OP_bit_piece, 64, 64)
; CHECK: ![[P3]] = !DIExpression(DW_OP_bit_piece, 192, 64)
; CHECK: ![[P4]] = !DIExpression(DW_OP_bit_piece, 256, 64)
; 
; struct p {
;   __SIZE_TYPE__ s;
;   __SIZE_TYPE__ t;
; };
;  
; struct r {
;   int i;
;   struct p x;
;   struct p y;
; };
;  
; extern int call_me(struct r);
; extern int maybe();
; extern __SIZE_TYPE__ t;
;  
; int test() {
;   if (maybe())
;     return 0;
;   struct p y = {t, t};
;   struct r r = {.y = y};
;   return call_me(r);
; }

; ModuleID = 'pr22393.cc'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

%struct.p = type { i64, i64 }
%struct.r = type { i32, %struct.p, %struct.p }

@t = external global i64

; Function Attrs: nounwind
define i32 @_Z4testv() #0 !dbg !17 {
entry:
  %retval = alloca i32, align 4
  %y = alloca %struct.p, align 8
  %r = alloca %struct.r, align 8
  %agg.tmp = alloca %struct.r, align 8
  %call = call i32 @_Z5maybev(), !dbg !24
  %tobool = icmp ne i32 %call, 0, !dbg !24
  br i1 %tobool, label %if.then, label %if.end, !dbg !26

if.then:                                          ; preds = %entry
  store i32 0, i32* %retval, !dbg !27
  br label %return, !dbg !27

if.end:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata %struct.p* %y, metadata !28, metadata !29), !dbg !30
  %s = getelementptr inbounds %struct.p, %struct.p* %y, i32 0, i32 0, !dbg !30
  %0 = load i64, i64* @t, align 8, !dbg !30
  store i64 %0, i64* %s, align 8, !dbg !30
  %t = getelementptr inbounds %struct.p, %struct.p* %y, i32 0, i32 1, !dbg !30
  %1 = load i64, i64* @t, align 8, !dbg !30
  store i64 %1, i64* %t, align 8, !dbg !30
  call void @llvm.dbg.declare(metadata %struct.r* %r, metadata !31, metadata !29), !dbg !32
  %i = getelementptr inbounds %struct.r, %struct.r* %r, i32 0, i32 0, !dbg !32
  store i32 0, i32* %i, align 4, !dbg !32
  %x = getelementptr inbounds %struct.r, %struct.r* %r, i32 0, i32 1, !dbg !32
  %s1 = getelementptr inbounds %struct.p, %struct.p* %x, i32 0, i32 0, !dbg !32
  store i64 0, i64* %s1, align 8, !dbg !32
  %t2 = getelementptr inbounds %struct.p, %struct.p* %x, i32 0, i32 1, !dbg !32
  store i64 0, i64* %t2, align 8, !dbg !32
  %y3 = getelementptr inbounds %struct.r, %struct.r* %r, i32 0, i32 2, !dbg !32
  %2 = bitcast %struct.p* %y3 to i8*, !dbg !32
  %3 = bitcast %struct.p* %y to i8*, !dbg !32
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %2, i8* %3, i64 16, i32 8, i1 false), !dbg !32
  %4 = bitcast %struct.r* %agg.tmp to i8*, !dbg !33
  %5 = bitcast %struct.r* %r to i8*, !dbg !33
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %4, i8* %5, i64 40, i32 8, i1 false), !dbg !33
  %call4 = call i32 @_Z7call_me1r(%struct.r* byval align 8 %agg.tmp), !dbg !33
  store i32 %call4, i32* %retval, !dbg !33
  br label %return, !dbg !33

return:                                           ; preds = %if.end, %if.then
  %6 = load i32, i32* %retval, !dbg !34
  ret i32 %6, !dbg !34
}

declare i32 @_Z5maybev()

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #3

declare i32 @_Z7call_me1r(%struct.r* byval align 8)

attributes #0 = { nounwind }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !22}
!llvm.ident = !{!23}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.7.0 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, subprograms: !16, globals: !2, imports: !2)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{}
!3 = !{!4, !10}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "p", line: 3, size: 128, align: 64, file: !5, elements: !6, identifier: "_ZTS1p")
!5 = !DIFile(filename: "pr22393.cc", directory: "")
!6 = !{!7, !9}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "s", line: 4, size: 64, align: 64, file: !5, scope: !"_ZTS1p", baseType: !8)
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "long unsigned int", size: 64, align: 64, encoding: DW_ATE_unsigned)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "t", line: 5, size: 64, align: 64, offset: 64, file: !5, scope: !"_ZTS1p", baseType: !8)
!10 = !DICompositeType(tag: DW_TAG_structure_type, name: "r", line: 8, size: 320, align: 64, file: !5, elements: !11, identifier: "_ZTS1r")
!11 = !{!12, !14, !15}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "i", line: 9, size: 32, align: 32, file: !5, scope: !"_ZTS1r", baseType: !13)
!13 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "x", line: 10, size: 128, align: 64, offset: 64, file: !5, scope: !"_ZTS1r", baseType: !"_ZTS1p")
!15 = !DIDerivedType(tag: DW_TAG_member, name: "y", line: 11, size: 128, align: 64, offset: 192, file: !5, scope: !"_ZTS1r", baseType: !"_ZTS1p")
!16 = !{!17}
!17 = distinct !DISubprogram(name: "test", linkageName: "_Z4testv", line: 18, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 18, file: !5, scope: !18, type: !19, variables: !2)
!18 = !DIFile(filename: "pr22393.cc", directory: "")
!19 = !DISubroutineType(types: !20)
!20 = !{!13}
!21 = !{i32 2, !"Dwarf Version", i32 4}
!22 = !{i32 2, !"Debug Info Version", i32 3}
!23 = !{!"clang version 3.7.0 "}
!24 = !DILocation(line: 19, scope: !25)
!25 = distinct !DILexicalBlock(line: 19, column: 0, file: !5, scope: !17)
!26 = !DILocation(line: 19, scope: !17)
!27 = !DILocation(line: 20, scope: !25)
!28 = !DILocalVariable(name: "y", line: 21, scope: !17, file: !18, type: !"_ZTS1p")
!29 = !DIExpression()
!30 = !DILocation(line: 21, scope: !17)
!31 = !DILocalVariable(name: "r", line: 22, scope: !17, file: !18, type: !"_ZTS1r")
!32 = !DILocation(line: 22, scope: !17)
!33 = !DILocation(line: 23, scope: !17)
!34 = !DILocation(line: 24, scope: !17)
