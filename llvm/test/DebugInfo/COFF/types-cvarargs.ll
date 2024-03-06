; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s
; RUN: llc < %s | llvm-mc -filetype=obj --triple=x86_64-windows | llvm-readobj - --codeview | FileCheck %s

; RUN: llc --try-experimental-debuginfo-iterators < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s
; RUN: llc --try-experimental-debuginfo-iterators < %s | llvm-mc -filetype=obj --triple=x86_64-windows | llvm-readobj - --codeview | FileCheck %s

; C++ source to regenerate:
; $ cat t.cpp
; typedef void (*FuncTypedef)(int, float, ...);
; FuncTypedef funcVar;
; namespace MemberTest {
;   class A {
;   public:
;     int MemberFunc(...) { return 1; }
;   };
; }
; int f () {
;   MemberTest::A v1;
;   v1.MemberFunc(1,20,0);
;   return 1;
; }
; $ clang t.cpp -S -emit-llvm -g -gcodeview -o t.ll

; CHECK:  MemberFuncId (0x100B) {
; CHECK:    TypeLeafKind: LF_MFUNC_ID (0x1602)
; CHECK:    ClassType: MemberTest::A (0x1003)
; CHECK:    FunctionType: int MemberTest::A::(<no type>) (0x1006)
; CHECK:    Name: MemberFunc
; CHECK:  }
; CHECK:  Subsection [
; CHECK:    SubSectionType: Symbols (0xF1)
; CHECK:    SubSectionSize:
; CHECK:    UDTSym {
; CHECK:      Kind: S_UDT (0x1108)
; CHECK:      Type: MemberTest::A (0x1008)
; CHECK:      UDTName: MemberTest::A
; CHECK:    }
; CHECK:    UDTSym {
; CHECK:      Kind: S_UDT (0x1108)
; CHECK:      Type: void (int, float, <no type>)* (0x100F)
; CHECK:      UDTName: FuncTypedef
; CHECK:    }
; CHECK:  ]

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.11.25507"

%"class.MemberTest::A" = type { i8 }

$"\01?MemberFunc@A@MemberTest@@QEAAHZZ" = comdat any

@"\01?funcVar@@3P6AXHMZZEA" = global ptr null, align 8, !dbg !0

; Function Attrs: noinline optnone uwtable
define i32 @"\01?f@@YAHXZ"() #0 !dbg !17 {
entry:
  %v1 = alloca %"class.MemberTest::A", align 1
  call void @llvm.dbg.declare(metadata ptr %v1, metadata !20, metadata !DIExpression()), !dbg !28
  %call = call i32 (ptr, ...) @"\01?MemberFunc@A@MemberTest@@QEAAHZZ"(ptr %v1, i32 1, i32 20, i64 0), !dbg !29
  ret i32 1, !dbg !30
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr i32 @"\01?MemberFunc@A@MemberTest@@QEAAHZZ"(ptr %this, ...) #2 comdat align 2 !dbg !31 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %this.addr, metadata !32, metadata !DIExpression()), !dbg !34
  %this1 = load ptr, ptr %this.addr, align 8
  ret i32 1, !dbg !35
}

attributes #0 = { noinline optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13, !14, !15}
!llvm.ident = !{!16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "funcVar", linkageName: "\01?funcVar@@3P6AXHMZZEA", scope: !2, file: !3, line: 4, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 7.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "t.cpp", directory: "D:\5Cupstream\5Cllvm\5Ctest\5CDebugInfo\5CCOFF", checksumkind: CSK_MD5, checksum: "d6582aff49f975763b736524db75f999")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_typedef, name: "FuncTypedef", file: !3, line: 3, baseType: !7)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10, !11, null}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!12 = !{i32 2, !"CodeView", i32 1}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 2}
!15 = !{i32 7, !"PIC Level", i32 2}
!16 = !{!"clang version 7.0.0 "}
!17 = distinct !DISubprogram(name: "f", linkageName: "\01?f@@YAHXZ", scope: !3, file: !3, line: 11, type: !18, isLocal: false, isDefinition: true, scopeLine: 11, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!18 = !DISubroutineType(types: !19)
!19 = !{!10}
!20 = !DILocalVariable(name: "v1", scope: !17, file: !3, line: 12, type: !21)
!21 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "A", scope: !22, file: !3, line: 6, size: 8, elements: !23, identifier: ".?AVA@MemberTest@@")
!22 = !DINamespace(name: "MemberTest", scope: null)
!23 = !{!24}
!24 = !DISubprogram(name: "MemberFunc", linkageName: "\01?MemberFunc@A@MemberTest@@QEAAHZZ", scope: !21, file: !3, line: 8, type: !25, isLocal: false, isDefinition: false, scopeLine: 8, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: false)
!25 = !DISubroutineType(types: !26)
!26 = !{!10, !27, null}
!27 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !21, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!28 = !DILocation(line: 12, column: 18, scope: !17)
!29 = !DILocation(line: 13, column: 7, scope: !17)
!30 = !DILocation(line: 14, column: 4, scope: !17)
!31 = distinct !DISubprogram(name: "MemberFunc", linkageName: "\01?MemberFunc@A@MemberTest@@QEAAHZZ", scope: !21, file: !3, line: 8, type: !25, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !2, declaration: !24, retainedNodes: !4)
!32 = !DILocalVariable(name: "this", arg: 1, scope: !31, type: !33, flags: DIFlagArtificial | DIFlagObjectPointer)
!33 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !21, size: 64)
!34 = !DILocation(line: 0, scope: !31)
!35 = !DILocation(line: 8, column: 28, scope: !31)
