; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s
; RUN: llc < %s | llvm-mc -filetype=obj --triple=x86_64-windows | llvm-readobj - --codeview | FileCheck %s

; RUN: llc --try-experimental-debuginfo-iterators < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s
; RUN: llc --try-experimental-debuginfo-iterators < %s | llvm-mc -filetype=obj --triple=x86_64-windows | llvm-readobj - --codeview | FileCheck %s

; C++ source to regenerate:
; struct A {
;   int NoRefQual();
;
;   int RefQual() &;
;   int RefQual() &&;
;
;   int LValueRef() &;
;
;   int RValueRef() &&;
; };
;
; void foo() {
;   A *GenericPtr = nullptr;
;   A a;
; }


; ModuleID = 'foo.cpp'
source_filename = "foo.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.15.26732"

%struct.A = type { i8 }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?foo@@YAXXZ"() #0 !dbg !10 {
entry:
  %GenericPtr = alloca ptr, align 8
  %a = alloca %struct.A, align 1
  call void @llvm.dbg.declare(metadata ptr %GenericPtr, metadata !13, metadata !DIExpression()), !dbg !28
  store ptr null, ptr %GenericPtr, align 8, !dbg !28
  call void @llvm.dbg.declare(metadata ptr %a, metadata !29, metadata !DIExpression()), !dbg !30
  ret void, !dbg !31
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.linker.options = !{!3, !4}
!llvm.module.flags = !{!5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "foo.cpp", directory: "D:\5C\5Csrc\5C\5Cllvmbuild\5C\5Cninja-x64", checksumkind: CSK_MD5, checksum: "d1b6ae9dc9ab85ca0a41c8b8c79a0b6a")
!2 = !{}
!3 = !{!"/DEFAULTLIB:libcmt.lib"}
!4 = !{!"/DEFAULTLIB:oldnames.lib"}
!5 = !{i32 2, !"CodeView", i32 1}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 2}
!8 = !{i32 7, !"PIC Level", i32 2}
!9 = !{!"clang version 8.0.0 "}
!10 = distinct !DISubprogram(name: "foo", linkageName: "?foo@@YAXXZ", scope: !1, file: !1, line: 12, type: !11, isLocal: false, isDefinition: true, scopeLine: 12, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!11 = !DISubroutineType(types: !12)
!12 = !{null}
!13 = !DILocalVariable(name: "GenericPtr", scope: !10, file: !1, line: 13, type: !14)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !1, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !16, identifier: ".?AUA@@")
!16 = !{!17, !22, !24, !26, !27}
!17 = !DISubprogram(name: "NoRefQual", linkageName: "?NoRefQual@A@@QEAAHXZ", scope: !15, file: !1, line: 2, type: !18, isLocal: false, isDefinition: false, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false)
!18 = !DISubroutineType(types: !19)
!19 = !{!20, !21}
!20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!22 = !DISubprogram(name: "RefQual", linkageName: "?RefQual@A@@QEGAAHXZ", scope: !15, file: !1, line: 4, type: !23, isLocal: false, isDefinition: false, scopeLine: 4, flags: DIFlagPrototyped | DIFlagLValueReference, isOptimized: false)
!23 = !DISubroutineType(flags: DIFlagLValueReference, types: !19)
!24 = !DISubprogram(name: "RefQual", linkageName: "?RefQual@A@@QEHAAHXZ", scope: !15, file: !1, line: 5, type: !25, isLocal: false, isDefinition: false, scopeLine: 5, flags: DIFlagPrototyped | DIFlagRValueReference, isOptimized: false)
!25 = !DISubroutineType(flags: DIFlagRValueReference, types: !19)
!26 = !DISubprogram(name: "LValueRef", linkageName: "?LValueRef@A@@QEGAAHXZ", scope: !15, file: !1, line: 7, type: !23, isLocal: false, isDefinition: false, scopeLine: 7, flags: DIFlagPrototyped | DIFlagLValueReference, isOptimized: false)
!27 = !DISubprogram(name: "RValueRef", linkageName: "?RValueRef@A@@QEHAAHXZ", scope: !15, file: !1, line: 9, type: !25, isLocal: false, isDefinition: false, scopeLine: 9, flags: DIFlagPrototyped | DIFlagRValueReference, isOptimized: false)
!28 = !DILocation(line: 13, scope: !10)
!29 = !DILocalVariable(name: "a", scope: !10, file: !1, line: 14, type: !15)
!30 = !DILocation(line: 14, scope: !10)
!31 = !DILocation(line: 15, scope: !10)




; CHECK: CodeViewTypes [
; CHECK:   Section: .debug$T (7)
; CHECK:   Magic: 0x4
; CHECK:   Pointer (0x1005) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: A (0x1003)
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: Pointer (0x0)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 1
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     IsRestrict: 0
; CHECK:     IsThisPtr&: 0
; CHECK:     IsThisPtr&&: 0
; CHECK:     SizeOf: 8
; CHECK:   }
; CHECK:   MemberFunction (0x1006) {
; CHECK:     TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK:     ReturnType: int (0x74)
; CHECK:     ClassType: A (0x1003)
; CHECK:     ThisType: A* const (0x1005)
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 0
; CHECK:     ArgListType: () (0x1000)
; CHECK:     ThisAdjustment: 0
; CHECK:   }
; CHECK:   Pointer (0x1007) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: A (0x1003)
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: Pointer (0x0)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 1
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     IsRestrict: 0
; CHECK:     IsThisPtr&: 1
; CHECK:     IsThisPtr&&: 0
; CHECK:     SizeOf: 136
; CHECK:   }
; CHECK:   MemberFunction (0x1008) {
; CHECK:     TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK:     ReturnType: int (0x74)
; CHECK:     ClassType: A (0x1003)
; CHECK:     ThisType: A* const (0x1007)
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 0
; CHECK:     ArgListType: () (0x1000)
; CHECK:     ThisAdjustment: 0
; CHECK:   }
; CHECK:   Pointer (0x1009) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: A (0x1003)
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: Pointer (0x0)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 1
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     IsRestrict: 0
; CHECK:     IsThisPtr&: 0
; CHECK:     IsThisPtr&&: 1
; CHECK:     SizeOf: 8
; CHECK:   }
; CHECK:   MemberFunction (0x100A) {
; CHECK:     TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK:     ReturnType: int (0x74)
; CHECK:     ClassType: A (0x1003)
; CHECK:     ThisType: A* const (0x1009)
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 0
; CHECK:     ArgListType: () (0x1000)
; CHECK:     ThisAdjustment: 0
; CHECK:   }
; CHECK:   MethodOverloadList (0x100B) {
; CHECK:     TypeLeafKind: LF_METHODLIST (0x1206)
; CHECK:     Method [
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: int A::() (0x1008)
; CHECK:     ]
; CHECK:     Method [
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: int A::() (0x100A)
; CHECK:     ]
; CHECK:   }
; CHECK:   FieldList (0x100C) {
; CHECK:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:     OneMethod {
; CHECK:       TypeLeafKind: LF_ONEMETHOD (0x1511)
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: int A::() (0x1006)
; CHECK:       Name: NoRefQual
; CHECK:     }
; CHECK:     OverloadedMethod {
; CHECK:       TypeLeafKind: LF_METHOD (0x150F)
; CHECK:       MethodCount: 0x2
; CHECK:       MethodListIndex: 0x100B
; CHECK:       Name: RefQual
; CHECK:     }
; CHECK:     OneMethod {
; CHECK:       TypeLeafKind: LF_ONEMETHOD (0x1511)
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: int A::() (0x1008)
; CHECK:       Name: LValueRef
; CHECK:     }
; CHECK:     OneMethod {
; CHECK:       TypeLeafKind: LF_ONEMETHOD (0x1511)
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: int A::() (0x100A)
; CHECK:       Name: RValueRef
; CHECK:     }
; CHECK:   }
; CHECK: ]
