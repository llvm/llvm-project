; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s

; C++ source to regenerate:
; $ cat t.cpp
; char multi_dim_arr[2][5][7];
;
; struct incomplete_struct(*p_incomplete_struct_arr)[3];
; struct incomplete_struct {
;   int s1;
; } incomplete_struct_arr[3];
;
; void foo(int x) {
;   int dyn_size_arr[x];
;   dyn_size_arr[0] = 0;
; }
;
; typedef const volatile int T_INT;
; T_INT typedef_arr[4] = {0};
; $ clang t.cpp -S -emit-llvm -g -gcodeview -o t.ll

; CHECK: CodeViewTypes [
; CHECK:   Section: .debug$T (6)
; CHECK:   Magic: 0x4
; CHECK:   ArgList (0x1000) {
; CHECK:     TypeLeafKind: LF_ARGLIST (0x1201)
; CHECK:     NumArgs: 1
; CHECK:     Arguments [
; CHECK:       ArgType: int (0x74)
; CHECK:     ]
; CHECK:   }
; CHECK:   Procedure (0x1001) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: void (0x3)
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 1
; CHECK:     ArgListType: (int) (0x1000)
; CHECK:   }
; CHECK:   FuncId (0x1002) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: void (int) (0x1001)
; CHECK:     Name: foo
; CHECK:   }
; CHECK:   Array (0x1003) {
; CHECK:     TypeLeafKind: LF_ARRAY (0x1503)
; CHECK:     ElementType: int (0x74)
; CHECK:     IndexType: unsigned long (0x22)
; CHECK:     SizeOf: 4
; CHECK:     Name: 
; CHECK:   }
; CHECK:   Pointer (0x1004) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: 0x1003
; CHECK:     PointerAttributes: 0x2A
; CHECK:     PtrType: Near32 (0xA)
; CHECK:     PtrMode: LValueReference (0x1)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     SizeOf: 0
; CHECK:   }
; CHECK:   Array (0x1005) {
; CHECK:     TypeLeafKind: LF_ARRAY (0x1503)
; CHECK:     ElementType: char (0x70)
; CHECK:     IndexType: unsigned long (0x22)
; CHECK:     SizeOf: 7
; CHECK:     Name: 
; CHECK:   }
; CHECK:   Array (0x1006) {
; CHECK:     TypeLeafKind: LF_ARRAY (0x1503)
; CHECK:     ElementType: 0x1005
; CHECK:     IndexType: unsigned long (0x22)
; CHECK:     SizeOf: 35
; CHECK:     Name: 
; CHECK:   }
; CHECK:   Array (0x1007) {
; CHECK:     TypeLeafKind: LF_ARRAY (0x1503)
; CHECK:     ElementType: 0x1006
; CHECK:     IndexType: unsigned long (0x22)
; CHECK:     SizeOf: 70
; CHECK:     Name: 
; CHECK:   }
; CHECK:   Struct (0x1008) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 0
; CHECK:     Properties [ (0x280)
; CHECK:       ForwardReference (0x80)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: 0x0
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 0
; CHECK:     Name: incomplete_struct
; CHECK:     LinkageName: .?AUincomplete_struct@@
; CHECK:   }
; CHECK:   Array (0x1009) {
; CHECK:     TypeLeafKind: LF_ARRAY (0x1503)
; CHECK:     ElementType: incomplete_struct (0x1008)
; CHECK:     IndexType: unsigned long (0x22)
; CHECK:     SizeOf: 12
; CHECK:     Name: 
; CHECK:   }
; CHECK:   Pointer (0x100A) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: 0x1009
; CHECK:     PointerAttributes: 0x800A
; CHECK:     PtrType: Near32 (0xA)
; CHECK:     PtrMode: Pointer (0x0)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     SizeOf: 4
; CHECK:   }
; CHECK:   FieldList (0x100B) {
; CHECK:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:     DataMember {
; CHECK:       TypeLeafKind: LF_MEMBER (0x150D)
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: int (0x74)
; CHECK:       FieldOffset: 0x0
; CHECK:       Name: s1
; CHECK:     }
; CHECK:   }
; CHECK:   Struct (0x100C) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 1
; CHECK:     Properties [ (0x200)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x100B)
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 4
; CHECK:     Name: incomplete_struct
; CHECK:     LinkageName: .?AUincomplete_struct@@
; CHECK:   }
; CHECK:   StringId (0x100D) {
; CHECK:     TypeLeafKind: LF_STRING_ID (0x1605)
; CHECK:     Id: 0x0
; CHECK:     StringData: \t.cpp
; CHECK:   }
; CHECK:   UdtSourceLine (0x100E) {
; CHECK:     TypeLeafKind: LF_UDT_SRC_LINE (0x1606)
; CHECK:     UDT: incomplete_struct (0x100C)
; CHECK:     SourceFile: \t.cpp (0x100D)
; CHECK:     LineNumber: 4
; CHECK:   }
; CHECK:   Modifier (0x100F) {
; CHECK:     TypeLeafKind: LF_MODIFIER (0x1001)
; CHECK:     ModifiedType: int (0x74)
; CHECK:     Modifiers [ (0x3)
; CHECK:       Const (0x1)
; CHECK:       Volatile (0x2)
; CHECK:     ]
; CHECK:   }
; CHECK:   Array (0x1010) {
; CHECK:     TypeLeafKind: LF_ARRAY (0x1503)
; CHECK:     ElementType: const volatile int (0x100F)
; CHECK:     IndexType: unsigned long (0x22)
; CHECK:     SizeOf: 16
; CHECK:     Name: 
; CHECK:   }
; CHECK: ]

source_filename = "test/DebugInfo/COFF/types-array-advanced.ll"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.31101"

%struct.incomplete_struct = type { i32 }

@"\01?multi_dim_arr@@3PAY146DA" = global [2 x [5 x [7 x i8]]] zeroinitializer, align 1, !dbg !0
@"\01?p_incomplete_struct_arr@@3PAY02Uincomplete_struct@@A" = global [3 x i8]* null, align 4, !dbg !6
@"\01?incomplete_struct_arr@@3PAUincomplete_struct@@A" = global [3 x %struct.incomplete_struct] zeroinitializer, align 4, !dbg !16
@"\01?typedef_arr@@3SDHD" = constant [4 x i32] zeroinitializer, align 4, !dbg !18

; Function Attrs: nounwind
define void @"\01?foo@@YAXH@Z"(i32 %x) #0 !dbg !35 {
entry:
  %x.addr = alloca i32, align 4
  %saved_stack = alloca i8*, align 4
  store i32 %x, i32* %x.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %x.addr, metadata !38, metadata !39), !dbg !40
  %0 = load i32, i32* %x.addr, align 4, !dbg !41
  %1 = call i8* @llvm.stacksave(), !dbg !42
  store i8* %1, i8** %saved_stack, align 4, !dbg !42
  %vla = alloca i32, i32 %0, align 4, !dbg !42
  call void @llvm.dbg.declare(metadata i32* %vla, metadata !43, metadata !47), !dbg !48
  %arrayidx = getelementptr inbounds i32, i32* %vla, i32 0, !dbg !49
  store i32 0, i32* %arrayidx, align 4, !dbg !50
  %2 = load i8*, i8** %saved_stack, align 4, !dbg !51
  call void @llvm.stackrestore(i8* %2), !dbg !51
  ret void, !dbg !51
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare i8* @llvm.stacksave() #0

; Function Attrs: nounwind
declare void @llvm.stackrestore(i8*) #0

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!32, !33}
!llvm.ident = !{!34}

!0 = distinct !DIGlobalVariableExpression(var: !1)
!1 = !DIGlobalVariable(name: "multi_dim_arr", linkageName: "\01?multi_dim_arr@@3PAY146DA", scope: !2, file: !3, line: 1, type: !26, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 3.9.0 (trunk 273874)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "t.cpp", directory: "/")
!4 = !{}
!5 = !{!0, !6, !16, !18}
!6 = distinct !DIGlobalVariableExpression(var: !7)
!7 = !DIGlobalVariable(name: "p_incomplete_struct_arr", linkageName: "\01?p_incomplete_struct_arr@@3PAY02Uincomplete_struct@@A", scope: !2, file: !3, line: 3, type: !8, isLocal: false, isDefinition: true)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 32, align: 32)
!9 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, elements: !14)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "incomplete_struct", file: !3, line: 4, size: 32, align: 32, elements: !11, identifier: ".?AUincomplete_struct@@")
!11 = !{!12}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "s1", scope: !10, file: !3, line: 5, baseType: !13, size: 32, align: 32)
!13 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!14 = !{!15}
!15 = !DISubrange(count: 3)
!16 = distinct !DIGlobalVariableExpression(var: !17)
!17 = !DIGlobalVariable(name: "incomplete_struct_arr", linkageName: "\01?incomplete_struct_arr@@3PAUincomplete_struct@@A", scope: !2, file: !3, line: 6, type: !9, isLocal: false, isDefinition: true)
!18 = distinct !DIGlobalVariableExpression(var: !19)
!19 = !DIGlobalVariable(name: "typedef_arr", linkageName: "\01?typedef_arr@@3SDHD", scope: !2, file: !3, line: 14, type: !20, isLocal: false, isDefinition: true)
!20 = !DICompositeType(tag: DW_TAG_array_type, baseType: !21, size: 128, align: 32, elements: !24)
!21 = !DIDerivedType(tag: DW_TAG_typedef, name: "T_INT", file: !3, line: 13, baseType: !22)
!22 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !23)
!23 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !13)
!24 = !{!25}
!25 = !DISubrange(count: 4)
!26 = !DICompositeType(tag: DW_TAG_array_type, baseType: !27, size: 560, align: 8, elements: !28)
!27 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!28 = !{!29, !30, !31}
!29 = !DISubrange(count: 2)
!30 = !DISubrange(count: 5)
!31 = !DISubrange(count: 7)
!32 = !{i32 2, !"CodeView", i32 1}
!33 = !{i32 2, !"Debug Info Version", i32 3}
!34 = !{!"clang version 3.9.0 (trunk 273874)"}
!35 = distinct !DISubprogram(name: "foo", linkageName: "\01?foo@@YAXH@Z", scope: !3, file: !3, line: 8, type: !36, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !2, variables: !4)
!36 = !DISubroutineType(types: !37)
!37 = !{null, !13}
!38 = !DILocalVariable(name: "x", arg: 1, scope: !35, file: !3, line: 8, type: !13)
!39 = !DIExpression()
!40 = !DILocation(line: 8, column: 14, scope: !35)
!41 = !DILocation(line: 9, column: 21, scope: !35)
!42 = !DILocation(line: 9, column: 4, scope: !35)
!43 = !DILocalVariable(name: "dyn_size_arr", scope: !35, file: !3, line: 9, type: !44)
!44 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, align: 32, elements: !45)
!45 = !{!46}
!46 = !DISubrange(count: -1)
!47 = !DIExpression(DW_OP_deref)
!48 = !DILocation(line: 9, column: 8, scope: !35)
!49 = !DILocation(line: 10, column: 4, scope: !35)
!50 = !DILocation(line: 10, column: 20, scope: !35)
!51 = !DILocation(line: 11, column: 1, scope: !35)

