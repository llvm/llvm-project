; RUN: opt -module-summary -o %t1.bc %s
; RUN: opt -module-summary -o %t2.bc %S/Inputs/dicompositetype-unique2.ll
; RUN: llvm-lto --thinlto-action=run %t1.bc %t2.bc -thinlto-save-temps=%t3.
; RUN: llvm-dis %t3.0.3.imported.bc -o - | FileCheck %s
; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t --save-temps \
; RUN: -r %t1.bc,_ZN1CD2Ev,pl \
; RUN: -r %t1.bc,_ZN4CFVSD2Ev,l \
; RUN: -r %t1.bc,_Z3Getv,l \
; RUN: -r %t2.bc,_ZN4CFVSD2Ev,pl \
; RUN: -r %t2.bc,_Z3Getv,l
; RUN: llvm-dis %t.1.3.import.bc -o - | FileCheck %s

; Only llvm-lto2 adds the dso_local keyword, hence the {{.*}}
; CHECK: define available_externally{{.*}} void @_ZN4CFVSD2Ev

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-scei-ps4"

%class.C = type <{ i32 (...)**, %class.A, %struct.CFVS, [6 x i8] }>
%class.A = type { %struct.Vec }
%struct.Vec = type { i8 }
%struct.CFVS = type { %struct.Vec }
%struct.S = type { i8 }

define void @_ZN1CD2Ev(%class.C* %this) unnamed_addr align 2 !dbg !8 {
entry:
  %this.addr = alloca %class.C*, align 8
  %this1 = load %class.C*, %class.C** %this.addr, align 8
  %m = getelementptr inbounds %class.C, %class.C* %this1, i32 0, i32 2
  call void @_ZN4CFVSD2Ev(%struct.CFVS* %m), !dbg !50
  ret void
}

declare void @_ZN4CFVSD2Ev(%struct.CFVS*) unnamed_addr

declare dereferenceable(1) %struct.S* @_Z3Getv()

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 6.0.0 (trunk 321360) (llvm/trunk 321359)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "bz188598-a.cpp", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!8 = distinct !DISubprogram(name: "~C", linkageName: "_ZN1CD2Ev", scope: !9, file: !1, line: 9, type: !47, isLocal: false, isDefinition: true, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !46, retainedNodes: !2)
!9 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "C", file: !1, line: 5, size: 128, elements: !10, vtableHolder: !9, identifier: "_ZTS1C")
!10 = !{!38, !46}
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Vec<&Get>", file: !16, line: 4, size: 8, elements: !17, templateParams: !22, identifier: "_ZTS3VecIXadL_Z3GetvEEE")
!16 = !DIFile(filename: "./bz188598.h", directory: ".")
!17 = !{!55}
!22 = !{!23}
!23 = !DITemplateValueParameter(name: "F", type: !24, value: %struct.S* ()* @_Z3Getv)
!24 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !25, size: 64)
!25 = !DIDerivedType(tag: DW_TAG_typedef, name: "Func", file: !16, line: 2, baseType: !26)
!26 = !DISubroutineType(types: !27)
!27 = !{!55}
!38 = !DIDerivedType(tag: DW_TAG_member, name: "m", scope: !9, file: !1, line: 7, baseType: !39, size: 8, offset: 72)
!39 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CFVS", file: !16, line: 7, size: 8, elements: !40, identifier: "_ZTS4CFVS")
!40 = !{!41}
!41 = !DIDerivedType(tag: DW_TAG_member, name: "m_val", scope: !39, file: !16, line: 9, baseType: !15, size: 8)
!46 = !DISubprogram(name: "~C", scope: !9, file: !1, line: 6, type: !47, isLocal: false, isDefinition: false, scopeLine: 6, containingType: !9, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 0, flags: DIFlagPrototyped, isOptimized: false)
!47 = !DISubroutineType(types: !48)
!48 = !{!55}
!50 = !DILocation(line: 9, scope: !51)
!51 = distinct !DILexicalBlock(scope: !8, file: !1, line: 9)
!55 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
