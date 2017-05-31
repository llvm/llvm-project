; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s

; Check that we keep namespace scopes around the same way MSVC does.
; We do function scopes slightly differently, but everything should be alright.

; C++ source to regenerate:
; namespace foo {
; namespace bar {
; void baz() {
;   struct LocalRecord {
;     int x;
;   } l;
; };
; struct GlobalRecord {
;   int x;
;   void method();
; } g;
; void GlobalRecord::method() {}
; }
; }

; CHECK-LABEL:  FuncId ({{.*}}) {
; CHECK-NEXT:    TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK-NEXT:    ParentScope: foo::bar ({{.*}})
; CHECK-NEXT:    FunctionType: void () ({{.*}})
; CHECK-NEXT:    Name: baz
; CHECK-NEXT:  }

; CHECK:  Struct ({{.*}}) {
; CHECK:    TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:    MemberCount: 0
; CHECK:    Properties [ (0x180)
; CHECK:      ForwardReference (0x80)
; CHECK:      Scoped (0x100)
; CHECK:    ]
; CHECK:    FieldList: 0x0
; CHECK:    DerivedFrom: 0x0
; CHECK:    VShape: 0x0
; CHECK:    SizeOf: 0
; CHECK:    Name: foo::bar::baz::LocalRecord
; CHECK:  }

; CHECK:  Struct ({{.*}}) {
; CHECK:    TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:    MemberCount: 1
; CHECK:    Properties [ (0x100)
; CHECK:      Scoped (0x100)
; CHECK:    ]
; CHECK:    Name: foo::bar::baz::LocalRecord
; CHECK:  }

; CHECK:  Struct ({{.*}}) {
; CHECK:    TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:    MemberCount: 0
; CHECK:    Properties [ (0x280)
; CHECK:      ForwardReference (0x80)
; CHECK:      HasUniqueName (0x200)
; CHECK:    ]
; CHECK:    FieldList: 0x0
; CHECK:    DerivedFrom: 0x0
; CHECK:    VShape: 0x0
; CHECK:    SizeOf: 0
; CHECK:    Name: foo::bar::GlobalRecord
; CHECK:  }

; CHECK-LABEL: MemberFuncId ({{.*}}) {
; CHECK-NEXT:    TypeLeafKind: LF_MFUNC_ID (0x1602)
; CHECK-NEXT:    ClassType: foo::bar::GlobalRecord ({{.*}})
; CHECK-NEXT:    FunctionType: void foo::bar::GlobalRecord::() ({{.*}})
; CHECK-NEXT:    Name: method
; CHECK-NEXT:  }


; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.23918"

%"struct.foo::bar::GlobalRecord" = type { i32 }
%struct.LocalRecord = type { i32 }

@"\01?g@bar@foo@@3UGlobalRecord@12@A" = global %"struct.foo::bar::GlobalRecord" zeroinitializer, align 4

; Function Attrs: nounwind uwtable
define void @"\01?baz@bar@foo@@YAXXZ"() #0 !dbg !19 {
entry:
  %l = alloca %struct.LocalRecord, align 4
  call void @llvm.dbg.declare(metadata %struct.LocalRecord* %l, metadata !22, metadata !26), !dbg !27
  ret void, !dbg !28
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind uwtable
define void @"\01?method@GlobalRecord@bar@foo@@QEAAXXZ"(%"struct.foo::bar::GlobalRecord"* %this) #0 align 2 !dbg !29 {
entry:
  %this.addr = alloca %"struct.foo::bar::GlobalRecord"*, align 8
  store %"struct.foo::bar::GlobalRecord"* %this, %"struct.foo::bar::GlobalRecord"** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %"struct.foo::bar::GlobalRecord"** %this.addr, metadata !30, metadata !26), !dbg !32
  %this1 = load %"struct.foo::bar::GlobalRecord"*, %"struct.foo::bar::GlobalRecord"** %this.addr, align 8
  ret void, !dbg !33
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!15, !16, !17}
!llvm.ident = !{!18}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!2 = !{}
!3 = !{!4}
!4 = distinct !DIGlobalVariable(name: "g", linkageName: "\01?g@bar@foo@@3UGlobalRecord@12@A", scope: !5, file: !1, line: 12, type: !7, isLocal: false, isDefinition: true, variable: %"struct.foo::bar::GlobalRecord"* @"\01?g@bar@foo@@3UGlobalRecord@12@A")
!5 = !DINamespace(name: "bar", scope: !6, file: !1, line: 2)
!6 = !DINamespace(name: "foo", scope: null, file: !1, line: 1)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "GlobalRecord", scope: !5, file: !1, line: 9, size: 32, align: 32, elements: !8, identifier: ".?AUGlobalRecord@bar@foo@@")
!8 = !{!9, !11}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !7, file: !1, line: 10, baseType: !10, size: 32, align: 32)
!10 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !DISubprogram(name: "method", linkageName: "\01?method@GlobalRecord@bar@foo@@QEAAXXZ", scope: !7, file: !1, line: 11, type: !12, isLocal: false, isDefinition: false, scopeLine: 11, flags: DIFlagPrototyped, isOptimized: false)
!12 = !DISubroutineType(types: !13)
!13 = !{null, !14}
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!15 = !{i32 2, !"CodeView", i32 1}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = !{i32 1, !"PIC Level", i32 2}
!18 = !{!"clang version 3.9.0 "}
!19 = distinct !DISubprogram(name: "baz", linkageName: "\01?baz@bar@foo@@YAXXZ", scope: !5, file: !1, line: 3, type: !20, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!20 = !DISubroutineType(types: !21)
!21 = !{null}
!22 = !DILocalVariable(name: "l", scope: !19, file: !1, line: 6, type: !23)
!23 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "LocalRecord", scope: !19, file: !1, line: 4, size: 32, align: 32, elements: !24)
!24 = !{!25}
!25 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !23, file: !1, line: 5, baseType: !10, size: 32, align: 32)
!26 = !DIExpression()
!27 = !DILocation(line: 6, column: 5, scope: !19)
!28 = !DILocation(line: 7, column: 1, scope: !19)
!29 = distinct !DISubprogram(name: "method", linkageName: "\01?method@GlobalRecord@bar@foo@@QEAAXXZ", scope: !7, file: !1, line: 13, type: !12, isLocal: false, isDefinition: true, scopeLine: 13, flags: DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !11, variables: !2)
!30 = !DILocalVariable(name: "this", arg: 1, scope: !29, type: !31, flags: DIFlagArtificial | DIFlagObjectPointer)
!31 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, align: 64)
!32 = !DILocation(line: 0, scope: !29)
!33 = !DILocation(line: 13, column: 30, scope: !29)
