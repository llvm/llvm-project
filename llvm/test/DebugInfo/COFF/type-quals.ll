; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s
; RUN: llc < %s | llvm-mc -filetype=obj --triple=x86_64-windows | llvm-readobj - --codeview | FileCheck %s

; C++ source to regenerate:
; $ cat m.cpp
; union Union {
;   int * __restrict x_member;
;   float * __restrict y_member;
;   int* volatile __restrict m_volatile;
; };
;
; int f(const volatile int* __restrict arg_crv) {
;   Union u;
;   const int *p;
;   const volatile int v = 0;
;   return 1;
; }
;
; void g(int& __restrict arg_ref) {
;   const int x = 10;
;   const char str[] = "abc";
; }
;
; void h() {
;   struct Foo {
;     int a;
;     int func(int x) __restrict { return 1; }
;   };
;
;   Foo s = { 10 };
;
;   int* __restrict p_object = &s.a;
;
;   int Foo:: * __restrict p_data_member = &Foo::a;
;
;   int (Foo::* p_member_func)(int) __restrict = &Foo::func;
; }
;
; $ clang++ m.cpp -S -emit-llvm -g -gcodeview -o m.ll


; CHECK: CodeViewTypes [
; CHECK:   Section: .debug$T (6)
; CHECK:   Magic: 0x4
; CHECK:   Modifier (0x1000) {
; CHECK:     TypeLeafKind: LF_MODIFIER (0x1001)
; CHECK:     ModifiedType: int (0x74)
; CHECK:     Modifiers [ (0x3)
; CHECK:       Const (0x1)
; CHECK:       Volatile (0x2)
; CHECK:     ]
; CHECK:   }
; CHECK:   Pointer (0x1001) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: const volatile int (0x1000)
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: Pointer (0x0)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     IsRestrict: 1
; CHECK:     SizeOf: 8
; CHECK:   }
; CHECK:   ArgList (0x1002) {
; CHECK:     TypeLeafKind: LF_ARGLIST (0x1201)
; CHECK:     NumArgs: 1
; CHECK:     Arguments [
; CHECK:       ArgType: const volatile int* __restrict (0x1001)
; CHECK:     ]
; CHECK:   }
; CHECK:   Procedure (0x1003) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: int (0x74)
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 1
; CHECK:     ArgListType: (const volatile int* __restrict) (0x1002)
; CHECK:   }
; CHECK:   FuncId (0x1004) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: int (const volatile int* __restrict) (0x1003)
; CHECK:     Name: f
; CHECK:   }
; CHECK:   Union (0x1005) {
; CHECK:     TypeLeafKind: LF_UNION (0x1506)
; CHECK:     MemberCount: 0
; CHECK:     Properties [ (0x280)
; CHECK:       ForwardReference (0x80)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: 0x0
; CHECK:     SizeOf: 0
; CHECK:     Name: Union
; CHECK:     LinkageName: .?ATUnion@@
; CHECK:   }
; CHECK:   Pointer (0x1006) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: int (0x74)
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: Pointer (0x0)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     IsRestrict: 1
; CHECK:     SizeOf: 8
; CHECK:   }
; CHECK:   Pointer (0x1007) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: float (0x40)
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: Pointer (0x0)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     IsRestrict: 1
; CHECK:     SizeOf: 8
; CHECK:   }
; CHECK:   Pointer (0x1008) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: int (0x74)
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: Pointer (0x0)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 1
; CHECK:     IsUnaligned: 0
; CHECK:     IsRestrict: 1
; CHECK:     SizeOf: 8
; CHECK:   }
; CHECK:   FieldList (0x1009) {
; CHECK:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:     DataMember {
; CHECK:       TypeLeafKind: LF_MEMBER (0x150D)
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: int* __restrict (0x1006)
; CHECK:       FieldOffset: 0x0
; CHECK:       Name: x_member
; CHECK:     }
; CHECK:     DataMember {
; CHECK:       TypeLeafKind: LF_MEMBER (0x150D)
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: float* __restrict (0x1007)
; CHECK:       FieldOffset: 0x0
; CHECK:       Name: y_member
; CHECK:     }
; CHECK:     DataMember {
; CHECK:       TypeLeafKind: LF_MEMBER (0x150D)
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: int* volatile __restrict (0x1008)
; CHECK:       FieldOffset: 0x0
; CHECK:       Name: m_volatile
; CHECK:     }
; CHECK:   }

; CHECK:   Modifier (0x100D) {
; CHECK:     TypeLeafKind: LF_MODIFIER (0x1001)
; CHECK:     ModifiedType: int (0x74)
; CHECK:     Modifiers [ (0x1)
; CHECK:       Const (0x1)
; CHECK:     ]
; CHECK:   }
; CHECK:   Pointer (0x100E) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: const int (0x100D)
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: Pointer (0x0)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     IsRestrict: 0
; CHECK:     SizeOf: 8
; CHECK:   }
; CHECK:   Pointer (0x100F) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: int (0x74)
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: LValueReference (0x1)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     IsRestrict: 1
; CHECK:     SizeOf: 8
; CHECK:   }
; CHECK:   ArgList (0x1010) {
; CHECK:     TypeLeafKind: LF_ARGLIST (0x1201)
; CHECK:     NumArgs: 1
; CHECK:     Arguments [
; CHECK:       ArgType: int& __restrict (0x100F)
; CHECK:     ]
; CHECK:   }
; CHECK:   Procedure (0x1011) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: void (0x3)
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 1
; CHECK:     ArgListType: (int& __restrict) (0x1010)
; CHECK:   }
; CHECK:   FuncId (0x1012) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: void (int& __restrict) (0x1011)
; CHECK:     Name: g
; CHECK:   }
; CHECK:   Modifier (0x1013) {
; CHECK:     TypeLeafKind: LF_MODIFIER (0x1001)
; CHECK:     ModifiedType: char (0x70)
; CHECK:     Modifiers [ (0x1)
; CHECK:       Const (0x1)
; CHECK:     ]
; CHECK:   }
; CHECK:   Array (0x1014) {
; CHECK:     TypeLeafKind: LF_ARRAY (0x1503)
; CHECK:     ElementType: const char (0x1013)
; CHECK:     IndexType: unsigned __int64 (0x23)
; CHECK:     SizeOf: 4
; CHECK:     Name:
; CHECK:   }
; CHECK:   ArgList (0x1015) {
; CHECK:     TypeLeafKind: LF_ARGLIST (0x1201)
; CHECK:     NumArgs: 0
; CHECK:     Arguments [
; CHECK:     ]
; CHECK:   }
; CHECK:   Procedure (0x1016) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: void (0x3)
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 0
; CHECK:     ArgListType: () (0x1015)
; CHECK:   }
; CHECK:   FuncId (0x1017) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: void () (0x1016)
; CHECK:     Name: h
; CHECK:   }
; CHECK:   Struct (0x1018) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 0
; CHECK:     Properties [ (0x180)
; CHECK:       ForwardReference (0x80)
; CHECK:       Scoped (0x100)
; CHECK:     ]
; CHECK:     FieldList: 0x0
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 0
; CHECK:     Name: h::Foo
; CHECK:   }
; CHECK:   Pointer (0x1019) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: h::Foo (0x1018)
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: Pointer (0x0)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 1
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     IsRestrict: 0
; CHECK:     SizeOf: 8
; CHECK:   }
; CHECK:   ArgList (0x101A) {
; CHECK:     TypeLeafKind: LF_ARGLIST (0x1201)
; CHECK:     NumArgs: 1
; CHECK:     Arguments [
; CHECK:       ArgType: int (0x74)
; CHECK:     ]
; CHECK:   }
; CHECK:   MemberFunction (0x101B) {
; CHECK:     TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK:     ReturnType: int (0x74)
; CHECK:     ClassType: h::Foo (0x1018)
; CHECK:     ThisType: h::Foo* const (0x1019)
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 1
; CHECK:     ArgListType: (int) (0x101A)
; CHECK:     ThisAdjustment: 0
; CHECK:   }
; CHECK:   FieldList (0x101C) {
; CHECK:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:     DataMember {
; CHECK:       TypeLeafKind: LF_MEMBER (0x150D)
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: int (0x74)
; CHECK:       FieldOffset: 0x0
; CHECK:       Name: a
; CHECK:     }
; CHECK:     OneMethod {
; CHECK:       TypeLeafKind: LF_ONEMETHOD (0x1511)
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: int h::Foo::(int) (0x101B)
; CHECK:       Name: func
; CHECK:     }
; CHECK:   }
; CHECK:   Struct (0x101D) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 2
; CHECK:     Properties [ (0x100)
; CHECK:       Scoped (0x100)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x101C)
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 4
; CHECK:     Name: h::Foo
; CHECK:   }

; CHECK:   Pointer (0x101F) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: int (0x74)
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: PointerToDataMember (0x2)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     IsRestrict: 1
; CHECK:     SizeOf: 4
; CHECK:     ClassType: h::Foo (0x1018)
; CHECK:     Representation: SingleInheritanceData (0x1)
; CHECK:   }
; CHECK:   Pointer (0x1020) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: int h::Foo::(int) (0x101B)
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: PointerToMemberFunction (0x3)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     IsRestrict: 0
; CHECK:     SizeOf: 8
; CHECK:     ClassType: h::Foo (0x1018)
; CHECK:     Representation: SingleInheritanceFunction (0x5)
; CHECK:   }
; CHECK:   MemberFuncId (0x1021) {
; CHECK:     TypeLeafKind: LF_MFUNC_ID (0x1602)
; CHECK:     ClassType: h::Foo (0x1018)
; CHECK:     FunctionType: int h::Foo::(int) (0x101B)
; CHECK:     Name: func
; CHECK:   }
; CHECK:   Pointer (0x1022) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: h::Foo (0x1018)
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: Pointer (0x0)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     IsRestrict: 0
; CHECK:     SizeOf: 8
; CHECK:   }
; CHECK: ]

; CHECK-LABEL: CodeViewDebugInfo [
; CHECK-NEXT:   Section: .debug$S (5)
; CHECK:   Subsection [
; CHECK:     SubSectionType: Symbols (0xF1)
; CHECK:     GlobalProcIdSym {
; CHECK:       Kind: S_GPROC32_ID (0x1147)
; CHECK:       FunctionType: f ({{.*}})
; CHECK:       CodeOffset: ?f@@YAHPEIDH@Z+0x0
; CHECK:       DisplayName: f
; CHECK:       LinkageName: ?f@@YAHPEIDH@Z
; CHECK:     }
; CHECK:     LocalSym {
; CHECK:       Kind: S_LOCAL (0x113E)
; CHECK:       Type: const volatile int* __restrict (0x1001)
; CHECK:       VarName: arg_crv
; CHECK:     }
; CHECK:     LocalSym {
; CHECK:       Kind: S_LOCAL (0x113E)
; CHECK:       Type: Union (0x100A)
; CHECK:       VarName: u
; CHECK:     }
; CHECK:     LocalSym {
; CHECK:       Kind: S_LOCAL (0x113E)
; CHECK:       Type: const int* (0x100E)
; CHECK:       VarName: p
; CHECK:     }
; CHECK:     LocalSym {
; CHECK:       Kind: S_LOCAL (0x113E)
; CHECK:       Type: const volatile int (0x1000)
; CHECK:       VarName: v
; CHECK:     }

; ModuleID = 'm.cpp'
source_filename = "m.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.11.25507"

%struct.Foo = type { i32 }
%union.Union = type { i32* }

@"\01?str@?1??g@@YAXAEIAH@Z@3QBDB" = internal constant [4 x i8] c"abc\00", align 1, !dbg !0
@"\01?s@?1??h@@YAXXZ@3UFoo@?1??1@YAXXZ@A" = private unnamed_addr constant %struct.Foo { i32 10 }, align 4

; Function Attrs: noinline nounwind optnone uwtable
define i32 @"\01?f@@YAHPEIDH@Z"(i32* noalias %arg_crv) #0 !dbg !22 {
entry:
  %arg_crv.addr = alloca i32*, align 8
  %u = alloca %union.Union, align 8
  %p = alloca i32*, align 8
  %v = alloca i32, align 4
  store i32* %arg_crv, i32** %arg_crv.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %arg_crv.addr, metadata !29, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.declare(metadata %union.Union* %u, metadata !31, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.declare(metadata i32** %p, metadata !44, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.declare(metadata i32* %v, metadata !48, metadata !DIExpression()), !dbg !49
  store volatile i32 0, i32* %v, align 4, !dbg !49
  ret i32 1, !dbg !50
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone uwtable
define void @"\01?g@@YAXAEIAH@Z"(i32* noalias dereferenceable(4) %arg_ref) #0 !dbg !2 {
entry:
  %arg_ref.addr = alloca i32*, align 8
  %x = alloca i32, align 4
  store i32* %arg_ref, i32** %arg_ref.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %arg_ref.addr, metadata !51, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.declare(metadata i32* %x, metadata !53, metadata !DIExpression()), !dbg !54
  store i32 10, i32* %x, align 4, !dbg !54
  ret void, !dbg !55
}

; Function Attrs: noinline nounwind optnone uwtable
define void @"\01?h@@YAXXZ"() #0 !dbg !56 {
entry:
  %s = alloca %struct.Foo, align 4
  %p_object = alloca i32*, align 8
  %p_data_member = alloca i32, align 8
  %p_member_func = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata %struct.Foo* %s, metadata !59, metadata !DIExpression()), !dbg !68
  %0 = bitcast %struct.Foo* %s to i8*, !dbg !68
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %0, i8* align 4 bitcast (%struct.Foo* @"\01?s@?1??h@@YAXXZ@3UFoo@?1??1@YAXXZ@A" to i8*), i64 4, i1 false), !dbg !68
  call void @llvm.dbg.declare(metadata i32** %p_object, metadata !69, metadata !DIExpression()), !dbg !70
  %a = getelementptr inbounds %struct.Foo, %struct.Foo* %s, i32 0, i32 0, !dbg !71
  store i32* %a, i32** %p_object, align 8, !dbg !70
  call void @llvm.dbg.declare(metadata i32* %p_data_member, metadata !72, metadata !DIExpression()), !dbg !75
  store i32 0, i32* %p_data_member, align 8, !dbg !75
  call void @llvm.dbg.declare(metadata i8** %p_member_func, metadata !76, metadata !DIExpression()), !dbg !78
  store i8* bitcast (i32 (%struct.Foo*, i32)* @"\01?func@Foo@?1??h@@YAXXZ@QEIAAHH@Z" to i8*), i8** %p_member_func, align 8, !dbg !78
  ret void, !dbg !79
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #2

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @"\01?func@Foo@?1??h@@YAXXZ@QEIAAHH@Z"(%struct.Foo* %this, i32 %x) #0 align 2 !dbg !80 {
entry:
  %x.addr = alloca i32, align 4
  %this.addr = alloca %struct.Foo*, align 8
  store i32 %x, i32* %x.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %x.addr, metadata !81, metadata !DIExpression()), !dbg !82
  store %struct.Foo* %this, %struct.Foo** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.Foo** %this.addr, metadata !83, metadata !DIExpression()), !dbg !85
  %this1 = load %struct.Foo*, %struct.Foo** %this.addr, align 8
  ret i32 1, !dbg !86
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { argmemonly nounwind }

!llvm.dbg.cu = !{!9}
!llvm.module.flags = !{!17, !18, !19, !20}
!llvm.ident = !{!21}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "str", scope: !2, file: !3, line: 18, type: !12, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "g", linkageName: "\01?g@@YAXAEIAH@Z", scope: !3, file: !3, line: 16, type: !4, isLocal: false, isDefinition: true, scopeLine: 16, flags: DIFlagPrototyped, isOptimized: false, unit: !9, retainedNodes: !10)
!3 = !DIFile(filename: "m.cpp", directory: "C:\5CUsers\5CHui\5Ctmp\5Chui", checksumkind: CSK_MD5, checksum: "a8da0f4dca948db1ef1129c8728a881c")
!4 = !DISubroutineType(types: !5)
!5 = !{null, !6}
!6 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !7)
!7 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !8, size: 64)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 7.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !10, globals: !11)
!10 = !{}
!11 = !{!0}
!12 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 32, elements: !15)
!13 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !14)
!14 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!15 = !{!16}
!16 = !DISubrange(count: 4)
!17 = !{i32 2, !"CodeView", i32 1}
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{i32 1, !"wchar_size", i32 2}
!20 = !{i32 7, !"PIC Level", i32 2}
!21 = !{!"clang version 7.0.0 "}
!22 = distinct !DISubprogram(name: "f", linkageName: "\01?f@@YAHPEIDH@Z", scope: !3, file: !3, line: 9, type: !23, isLocal: false, isDefinition: true, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: false, unit: !9, retainedNodes: !10)
!23 = !DISubroutineType(types: !24)
!24 = !{!8, !25}
!25 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !26)
!26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !27, size: 64)
!27 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !28)
!28 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !8)
!29 = !DILocalVariable(name: "arg_crv", arg: 1, scope: !22, file: !3, line: 9, type: !25)
!30 = !DILocation(line: 9, column: 39, scope: !22)
!31 = !DILocalVariable(name: "u", scope: !22, file: !3, line: 10, type: !32)
!32 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "Union", file: !3, line: 3, size: 64, flags: DIFlagTypePassByValue, elements: !33, identifier: ".?ATUnion@@")
!33 = !{!34, !37, !41}
!34 = !DIDerivedType(tag: DW_TAG_member, name: "x_member", scope: !32, file: !3, line: 4, baseType: !35, size: 64)
!35 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !36)
!36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!37 = !DIDerivedType(tag: DW_TAG_member, name: "y_member", scope: !32, file: !3, line: 5, baseType: !38, size: 64)
!38 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !39)
!39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !40, size: 64)
!40 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!41 = !DIDerivedType(tag: DW_TAG_member, name: "m_volatile", scope: !32, file: !3, line: 6, baseType: !42, size: 64)
!42 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !35)
!43 = !DILocation(line: 10, column: 10, scope: !22)
!44 = !DILocalVariable(name: "p", scope: !22, file: !3, line: 11, type: !45)
!45 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !46, size: 64)
!46 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !8)
!47 = !DILocation(line: 11, column: 15, scope: !22)
!48 = !DILocalVariable(name: "v", scope: !22, file: !3, line: 12, type: !27)
!49 = !DILocation(line: 12, column: 23, scope: !22)
!50 = !DILocation(line: 13, column: 4, scope: !22)
!51 = !DILocalVariable(name: "arg_ref", arg: 1, scope: !2, file: !3, line: 16, type: !6)
!52 = !DILocation(line: 16, column: 25, scope: !2)
!53 = !DILocalVariable(name: "x", scope: !2, file: !3, line: 17, type: !46)
!54 = !DILocation(line: 17, column: 14, scope: !2)
!55 = !DILocation(line: 19, column: 2, scope: !2)
!56 = distinct !DISubprogram(name: "h", linkageName: "\01?h@@YAXXZ", scope: !3, file: !3, line: 21, type: !57, isLocal: false, isDefinition: true, scopeLine: 21, flags: DIFlagPrototyped, isOptimized: false, unit: !9, retainedNodes: !10)
!57 = !DISubroutineType(types: !58)
!58 = !{null}
!59 = !DILocalVariable(name: "s", scope: !56, file: !3, line: 27, type: !60)
!60 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", scope: !56, file: !3, line: 22, size: 32, flags: DIFlagTypePassByValue, elements: !61)
!61 = !{!62, !63}
!62 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !60, file: !3, line: 23, baseType: !8, size: 32)
!63 = !DISubprogram(name: "func", scope: !60, file: !3, line: 24, type: !64, isLocal: false, isDefinition: false, scopeLine: 24, flags: DIFlagPrototyped, isOptimized: false)
!64 = !DISubroutineType(types: !65)
!65 = !{!8, !66, !8}

; FIXME: Clang emits wrong debug info here because of PR17747. We should
; regenerate this IR when it is fixed.
!66 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !67, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!67 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !60)

!68 = !DILocation(line: 27, column: 8, scope: !56)
!69 = !DILocalVariable(name: "p_object", scope: !56, file: !3, line: 29, type: !35)
!70 = !DILocation(line: 29, column: 20, scope: !56)
!71 = !DILocation(line: 29, column: 34, scope: !56)
!72 = !DILocalVariable(name: "p_data_member", scope: !56, file: !3, line: 31, type: !73)
!73 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !74)
!74 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !8, size: 32, flags: DIFlagSingleInheritance, extraData: !60)
!75 = !DILocation(line: 31, column: 27, scope: !56)
!76 = !DILocalVariable(name: "p_member_func", scope: !56, file: !3, line: 33, type: !77)
!77 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !64, size: 64, flags: DIFlagSingleInheritance, extraData: !60)
!78 = !DILocation(line: 33, column: 16, scope: !56)
!79 = !DILocation(line: 34, column: 2, scope: !56)
!80 = distinct !DISubprogram(name: "func", linkageName: "\01?func@Foo@?1??h@@YAXXZ@QEIAAHH@Z", scope: !60, file: !3, line: 24, type: !64, isLocal: true, isDefinition: true, scopeLine: 24, flags: DIFlagPrototyped, isOptimized: false, unit: !9, declaration: !63, retainedNodes: !10)
!81 = !DILocalVariable(name: "x", arg: 2, scope: !80, file: !3, line: 24, type: !8)
!82 = !DILocation(line: 24, column: 19, scope: !80)
!83 = !DILocalVariable(name: "this", arg: 1, scope: !80, type: !84, flags: DIFlagArtificial | DIFlagObjectPointer)
!84 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !67, size: 64)
!85 = !DILocation(line: 0, scope: !80)
!86 = !DILocation(line: 24, column: 35, scope: !80)
