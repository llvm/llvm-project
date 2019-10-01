; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s
source_filename = "test/DebugInfo/COFF/purge-typedef-udts.ll"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc19.11.25506"

; C++ source to regenerate:
; $ cat t.cpp
; struct Foo;
; struct Bar {
;   Bar() {}
;   int X;
; };
;
; typedef Foo FooTypedef;
; typedef Bar BarTypedef;
;
; int func(void *F) { return 7; }
; int func(const FooTypedef *F) { return func((void*)F); }
; int func(const BarTypedef *B) { return func((void*)B->X); }

; CHECK-NOT: UDTName: FooTypedef
; CHECK: UDTName: BarTypedef

%struct.Foo = type opaque
%struct.Bar = type { i32 }

; Function Attrs: noinline nounwind optnone
define i32 @"\01?func@@YAHPAX@Z"(i8* %F) #0 !dbg !10 {
entry:
  %F.addr = alloca i8*, align 4
  store i8* %F, i8** %F.addr, align 4
  call void @llvm.dbg.declare(metadata i8** %F.addr, metadata !14, metadata !DIExpression()), !dbg !15
  ret i32 7, !dbg !16
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone
define i32 @"\01?func@@YAHPBUFoo@@@Z"(%struct.Foo* %F) #0 !dbg !17 {
entry:
  %F.addr = alloca %struct.Foo*, align 4
  store %struct.Foo* %F, %struct.Foo** %F.addr, align 4
  call void @llvm.dbg.declare(metadata %struct.Foo** %F.addr, metadata !24, metadata !DIExpression()), !dbg !25
  %0 = load %struct.Foo*, %struct.Foo** %F.addr, align 4, !dbg !26
  %1 = bitcast %struct.Foo* %0 to i8*, !dbg !26
  %call = call i32 @"\01?func@@YAHPAX@Z"(i8* %1), !dbg !27
  ret i32 %call, !dbg !28
}

; Function Attrs: noinline nounwind optnone
define i32 @"\01?func@@YAHPBUBar@@@Z"(%struct.Bar* %B) #0 !dbg !29 {
entry:
  %B.addr = alloca %struct.Bar*, align 4
  store %struct.Bar* %B, %struct.Bar** %B.addr, align 4
  call void @llvm.dbg.declare(metadata %struct.Bar** %B.addr, metadata !42, metadata !DIExpression()), !dbg !43
  %0 = load %struct.Bar*, %struct.Bar** %B.addr, align 4, !dbg !44
  %X = getelementptr inbounds %struct.Bar, %struct.Bar* %0, i32 0, i32 0, !dbg !45
  %1 = load i32, i32* %X, align 4, !dbg !45
  %2 = inttoptr i32 %1 to i8*, !dbg !46
  %call = call i32 @"\01?func@@YAHPAX@Z"(i8* %2), !dbg !47
  ret i32 %call, !dbg !48
}

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 6.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3)
!1 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvmbuild\5Cninja", checksumkind: CSK_MD5, checksum: "27c44c8a5531845f61f582a24ef5c151")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 32)
!5 = !{i32 1, !"NumRegisterParameters", i32 0}
!6 = !{i32 2, !"CodeView", i32 1}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 2}
!9 = !{!"clang version 6.0.0 "}
!10 = distinct !DISubprogram(name: "func", linkageName: "\01?func@@YAHPAX@Z", scope: !1, file: !1, line: 10, type: !11, isLocal: false, isDefinition: true, scopeLine: 10, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !4}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DILocalVariable(name: "F", arg: 1, scope: !10, file: !1, line: 10, type: !4)
!15 = !DILocation(line: 10, column: 16, scope: !10)
!16 = !DILocation(line: 10, column: 21, scope: !10)
!17 = distinct !DISubprogram(name: "func", linkageName: "\01?func@@YAHPBUFoo@@@Z", scope: !1, file: !1, line: 11, type: !18, isLocal: false, isDefinition: true, scopeLine: 11, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!18 = !DISubroutineType(types: !19)
!19 = !{!13, !20}
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !21, size: 32)
!21 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !22)
!22 = !DIDerivedType(tag: DW_TAG_typedef, name: "FooTypedef", file: !1, line: 7, baseType: !23)
!23 = !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !1, line: 1, flags: DIFlagFwdDecl, identifier: ".?AUFoo@@")
!24 = !DILocalVariable(name: "F", arg: 1, scope: !17, file: !1, line: 11, type: !20)
!25 = !DILocation(line: 11, column: 28, scope: !17)
!26 = !DILocation(line: 11, column: 52, scope: !17)
!27 = !DILocation(line: 11, column: 40, scope: !17)
!28 = !DILocation(line: 11, column: 33, scope: !17)
!29 = distinct !DISubprogram(name: "func", linkageName: "\01?func@@YAHPBUBar@@@Z", scope: !1, file: !1, line: 12, type: !30, isLocal: false, isDefinition: true, scopeLine: 12, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!30 = !DISubroutineType(types: !31)
!31 = !{!13, !32}
!32 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !33, size: 32)
!33 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !34)
!34 = !DIDerivedType(tag: DW_TAG_typedef, name: "BarTypedef", file: !1, line: 8, baseType: !35)
!35 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Bar", file: !1, line: 2, size: 32, elements: !36, identifier: ".?AUBar@@")
!36 = !{!37, !38}
!37 = !DIDerivedType(tag: DW_TAG_member, name: "X", scope: !35, file: !1, line: 4, baseType: !13, size: 32)
!38 = !DISubprogram(name: "Bar", scope: !35, file: !1, line: 3, type: !39, isLocal: false, isDefinition: false, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false)
!39 = !DISubroutineType(cc: DW_CC_BORLAND_thiscall, types: !40)
!40 = !{null, !41}
!41 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !35, size: 32, flags: DIFlagArtificial | DIFlagObjectPointer)
!42 = !DILocalVariable(name: "B", arg: 1, scope: !29, file: !1, line: 12, type: !32)
!43 = !DILocation(line: 12, column: 28, scope: !29)
!44 = !DILocation(line: 12, column: 52, scope: !29)
!45 = !DILocation(line: 12, column: 55, scope: !29)
!46 = !DILocation(line: 12, column: 45, scope: !29)
!47 = !DILocation(line: 12, column: 40, scope: !29)
!48 = !DILocation(line: 12, column: 33, scope: !29)
