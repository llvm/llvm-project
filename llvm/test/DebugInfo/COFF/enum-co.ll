; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s
; RUN: llc < %s | llvm-mc -filetype=obj --triple=x86_64-windows | llvm-readobj - --codeview | FileCheck %s

; Command to generate enum-co.ll
; $ clang++ enum-co.cpp -S -emit-llvm -g -gcodeview -o enum-co.ll
;
;
; #define DEFINE_FUNCTION(T) \
;   T Func_##T(T &arg) { return arg; };
;
; enum Enum { ON, OFF };  // Expect: CO = HasUniqueName
; Enum Func_Enum(Enum &arg) { return arg; }
;
; enum class EnumClass { RED, BLUE, NOTCARE }; // Expect: CO = HasUniqueName
; EnumClass Func_EnumClass(EnumClass &arg) { return arg; }
;
; void Func() {
;   enum ScopedEnum { ON, OFF }; // Expected: CO = HasUniqueName | Scoped
;   ScopedEnum SE;
;
;   struct Struct {
;     union Union {
;       enum NestedEnum { RED, BLUE }; // Expected: CO = HasUniqueName | Nested
;     };
;     Union U;
;   };
;   Struct S;
; }

; CHECK: Format: COFF-x86-64
; CHECK: Arch: x86_64
; CHECK: AddressSize: 64bit
; CHECK: CodeViewTypes [
; CHECK:   Section: .debug$T (6)
; CHECK:   Magic: 0x4
; CHECK:   Enum ({{.*}}) {
; CHECK:     TypeLeafKind: LF_ENUM (0x1507)
; CHECK:     NumEnumerators: 2
; CHECK:     Properties [ (0x200)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     UnderlyingType: int (0x74)
; CHECK:     FieldListType: <field list> ({{.*}})
; CHECK:     Name: Enum
; CHECK:     LinkageName: .?AW4Enum@@
; CHECK:   }
; CHECK:   Enum ({{.*}}) {
; CHECK:     TypeLeafKind: LF_ENUM (0x1507)
; CHECK:     NumEnumerators: 3
; CHECK:     Properties [ (0x200)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     UnderlyingType: int (0x74)
; CHECK:     FieldListType: <field list> ({{.*}})
; CHECK:     Name: EnumClass
; CHECK:     LinkageName: .?AW4EnumClass@@
; CHECK:   }
; CHECK:   Enum ({{.*}}) {
; CHECK:     TypeLeafKind: LF_ENUM (0x1507)
; CHECK:     NumEnumerators: 2
; CHECK:     Properties [ (0x300)
; CHECK:       HasUniqueName (0x200)
; CHECK:       Scoped (0x100)
; CHECK:     ]
; CHECK:     UnderlyingType: int (0x74)
; CHECK:     FieldListType: <field list> ({{.*}})
; CHECK:     Func::ScopedEnum
; CHECK:     LinkageName: .?AW4ScopedEnum@?1??Func@@YAXXZ@
; CHECK:   }
; CHECK:   Enum ({{.*}}) {
; CHECK:     TypeLeafKind: LF_ENUM (0x1507)
; CHECK:     NumEnumerators: 2
; CHECK:     Properties [ (0x208)
; CHECK        HasUniqueName (0x200)
; CHECK        Nested (0x8)
; CHECK:     ]
; CHECK:     UnderlyingType: int (0x74)
; CHECK:     FieldListType: <field list> ({{.*}})
; CHECK:     Name: Func::Struct::Union::NestedEnum
; CHECK:     LinkageName: .?AW4NestedEnum@Union@Struct@?1??Func@@YAXXZ@
; CHECK:   }


; ModuleID = 'enum-co.cpp'
source_filename = "enum-co.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.15.26729"

%struct.Struct = type { %"union.Func()::Struct::Union" }
%"union.Func()::Struct::Union" = type { i8 }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @"?Func_Enum@@YA?AW4Enum@@AEAW41@@Z"(i32* dereferenceable(4) %arg) #0 !dbg !30 {
entry:
  %arg.addr = alloca i32*, align 8
  store i32* %arg, i32** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %arg.addr, metadata !34, metadata !DIExpression()), !dbg !35
  %0 = load i32*, i32** %arg.addr, align 8, !dbg !35
  %1 = load i32, i32* %0, align 4, !dbg !35
  ret i32 %1, !dbg !35
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @"?Func_EnumClass@@YA?AW4EnumClass@@AEAW41@@Z"(i32* dereferenceable(4) %arg) #0 !dbg !36 {
entry:
  %arg.addr = alloca i32*, align 8
  store i32* %arg, i32** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %arg.addr, metadata !40, metadata !DIExpression()), !dbg !41
  %0 = load i32*, i32** %arg.addr, align 8, !dbg !41
  %1 = load i32, i32* %0, align 4, !dbg !41
  ret i32 %1, !dbg !41
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?Func@@YAXXZ"() #0 !dbg !14 {
entry:
  %SE = alloca i32, align 4
  %S = alloca %struct.Struct, align 1
  call void @llvm.dbg.declare(metadata i32* %SE, metadata !42, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.declare(metadata %struct.Struct* %S, metadata !44, metadata !DIExpression()), !dbg !45
  ret void, !dbg !46
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!25, !26, !27, !28}
!llvm.ident = !{!29}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "enum-co.cpp", directory: "D:\5Cupstream\5Cllvm\5Ctest\5CDebugInfo\5CCOFF", checksumkind: CSK_MD5, checksum: "2e53b90441669acca735bad28ed3a1ab")
!2 = !{!3, !8, !13, !18}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Enum", file: !1, line: 4, baseType: !4, size: 32, elements: !5, identifier: ".?AW4Enum@@")
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = !{!6, !7}
!6 = !DIEnumerator(name: "ON", value: 0)
!7 = !DIEnumerator(name: "OFF", value: 1)
!8 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "EnumClass", file: !1, line: 7, baseType: !4, size: 32, flags: DIFlagEnumClass, elements: !9, identifier: ".?AW4EnumClass@@")
!9 = !{!10, !11, !12}
!10 = !DIEnumerator(name: "RED", value: 0)
!11 = !DIEnumerator(name: "BLUE", value: 1)
!12 = !DIEnumerator(name: "NOTCARE", value: 2)
!13 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "ScopedEnum", scope: !14, file: !1, line: 11, baseType: !4, size: 32, elements: !5, identifier: ".?AW4ScopedEnum@?1??Func@@YAXXZ@")
!14 = distinct !DISubprogram(name: "Func", linkageName: "?Func@@YAXXZ", scope: !1, file: !1, line: 10, type: !15, isLocal: false, isDefinition: true, scopeLine: 10, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !17)
!15 = !DISubroutineType(types: !16)
!16 = !{null}
!17 = !{}
!18 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "NestedEnum", scope: !19, file: !1, line: 16, baseType: !4, size: 32, elements: !24, identifier: ".?AW4NestedEnum@Union@Struct@?1??Func@@YAXXZ@")
!19 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "Union", scope: !20, file: !1, line: 15, size: 8, flags: DIFlagTypePassByValue, elements: !23, identifier: ".?ATUnion@Struct@?1??Func@@YAXXZ@")
!20 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Struct", scope: !14, file: !1, line: 14, size: 8, flags: DIFlagTypePassByValue, elements: !21, identifier: ".?AUStruct@?1??Func@@YAXXZ@")
!21 = !{!19, !22}
!22 = !DIDerivedType(tag: DW_TAG_member, name: "U", scope: !20, file: !1, line: 18, baseType: !19, size: 8)
!23 = !{!18}
!24 = !{!10, !11}
!25 = !{i32 2, !"CodeView", i32 1}
!26 = !{i32 2, !"Debug Info Version", i32 3}
!27 = !{i32 1, !"wchar_size", i32 2}
!28 = !{i32 7, !"PIC Level", i32 2}
!29 = !{!"clang version 8.0.0"}
!30 = distinct !DISubprogram(name: "Func_Enum", linkageName: "?Func_Enum@@YA?AW4Enum@@AEAW41@@Z", scope: !1, file: !1, line: 5, type: !31, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !17)
!31 = !DISubroutineType(types: !32)
!32 = !{!3, !33}
!33 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !3, size: 64)
!34 = !DILocalVariable(name: "arg", arg: 1, scope: !30, file: !1, line: 5, type: !33)
!35 = !DILocation(line: 5, scope: !30)
!36 = distinct !DISubprogram(name: "Func_EnumClass", linkageName: "?Func_EnumClass@@YA?AW4EnumClass@@AEAW41@@Z", scope: !1, file: !1, line: 8, type: !37, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !17)
!37 = !DISubroutineType(types: !38)
!38 = !{!8, !39}
!39 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !8, size: 64)
!40 = !DILocalVariable(name: "arg", arg: 1, scope: !36, file: !1, line: 8, type: !39)
!41 = !DILocation(line: 8, scope: !36)
!42 = !DILocalVariable(name: "SE", scope: !14, file: !1, line: 12, type: !13)
!43 = !DILocation(line: 12, scope: !14)
!44 = !DILocalVariable(name: "S", scope: !14, file: !1, line: 20, type: !20)
!45 = !DILocation(line: 20, scope: !14)
!46 = !DILocation(line: 21, scope: !14, isImplicitCode: true)
