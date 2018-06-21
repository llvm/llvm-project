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

@"\01?g@bar@foo@@3UGlobalRecord@12@A" = global %"struct.foo::bar::GlobalRecord" zeroinitializer, align 4, !dbg !0

; Function Attrs: nounwind uwtable
define void @"\01?baz@bar@foo@@YAXXZ"() #0 !dbg !20 {
entry:
  %l = alloca %struct.LocalRecord, align 4
  call void @llvm.dbg.declare(metadata %struct.LocalRecord* %l, metadata !23, metadata !27), !dbg !28
  ret void, !dbg !29
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind uwtable
define void @"\01?method@GlobalRecord@bar@foo@@QEAAXXZ"(%"struct.foo::bar::GlobalRecord"* %this) #0 align 2 !dbg !30 {
entry:
  %this.addr = alloca %"struct.foo::bar::GlobalRecord"*, align 8
  store %"struct.foo::bar::GlobalRecord"* %this, %"struct.foo::bar::GlobalRecord"** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %"struct.foo::bar::GlobalRecord"** %this.addr, metadata !31, metadata !27), !dbg !33
  %this1 = load %"struct.foo::bar::GlobalRecord"*, %"struct.foo::bar::GlobalRecord"** %this.addr, align 8
  ret void, !dbg !34
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!13}
!llvm.module.flags = !{!16, !17, !18}
!llvm.ident = !{!19}

!0 = distinct !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "g", linkageName: "\01?g@bar@foo@@3UGlobalRecord@12@A", scope: !2, file: !3, line: 12, type: !5, isLocal: false, isDefinition: true)
!2 = !DINamespace(name: "bar", scope: !4)
!3 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!4 = !DINamespace(name: "foo", scope: null)
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "GlobalRecord", scope: !2, file: !3, line: 9, size: 32, align: 32, elements: !6, identifier: ".?AUGlobalRecord@bar@foo@@")
!6 = !{!7, !9}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !5, file: !3, line: 10, baseType: !8, size: 32, align: 32)
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DISubprogram(name: "method", linkageName: "\01?method@GlobalRecord@bar@foo@@QEAAXXZ", scope: !5, file: !3, line: 11, type: !10, isLocal: false, isDefinition: false, scopeLine: 11, flags: DIFlagPrototyped, isOptimized: false)
!10 = !DISubroutineType(types: !11)
!11 = !{null, !12}
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!13 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 3.9.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !14, globals: !15)
!14 = !{}
!15 = !{!0}
!16 = !{i32 2, !"CodeView", i32 1}
!17 = !{i32 2, !"Debug Info Version", i32 3}
!18 = !{i32 1, !"PIC Level", i32 2}
!19 = !{!"clang version 3.9.0 "}
!20 = distinct !DISubprogram(name: "baz", linkageName: "\01?baz@bar@foo@@YAXXZ", scope: !2, file: !3, line: 3, type: !21, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !13, variables: !14)
!21 = !DISubroutineType(types: !22)
!22 = !{null}
!23 = !DILocalVariable(name: "l", scope: !20, file: !3, line: 6, type: !24)
!24 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "LocalRecord", scope: !20, file: !3, line: 4, size: 32, align: 32, elements: !25)
!25 = !{!26}
!26 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !24, file: !3, line: 5, baseType: !8, size: 32, align: 32)
!27 = !DIExpression()
!28 = !DILocation(line: 6, column: 5, scope: !20)
!29 = !DILocation(line: 7, column: 1, scope: !20)
!30 = distinct !DISubprogram(name: "method", linkageName: "\01?method@GlobalRecord@bar@foo@@QEAAXXZ", scope: !5, file: !3, line: 13, type: !10, isLocal: false, isDefinition: true, scopeLine: 13, flags: DIFlagPrototyped, isOptimized: false, unit: !13, declaration: !9, variables: !14)
!31 = !DILocalVariable(name: "this", arg: 1, scope: !30, type: !32, flags: DIFlagArtificial | DIFlagObjectPointer)
!32 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64, align: 64)
!33 = !DILocation(line: 0, scope: !30)
!34 = !DILocation(line: 13, column: 30, scope: !30)

