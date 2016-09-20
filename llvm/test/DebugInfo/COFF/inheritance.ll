; RUN: llc < %s -filetype=obj -o - | llvm-readobj - -codeview | FileCheck %s

; C++ source to regenerate:
; $ cat t.cpp
; struct A { int a; };
; struct B : virtual A { int b; };
; struct C : virtual A { int c; };
; struct D : B, C {
;   virtual void f(); // make vbptr not offset zero
;   int d;
; };
; D d;
; $ clang -fno-rtti -g -gcodeview t.cpp -emit-llvm -S -o t.ll -O1

; D's field list comes first.
; CHECK:        FieldList ({{.*}}) {
; CHECK-NEXT:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:     BaseClass {
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       BaseType: B ({{.*}})
; CHECK-NEXT:       BaseOffset: 0x8
; CHECK-NEXT:     }
; CHECK-NEXT:     BaseClass {
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       BaseType: C ({{.*}})
; CHECK-NEXT:       BaseOffset: 0x18
; CHECK-NEXT:     }
; CHECK:        }

; Then B's field list.
; CHECK:        FieldList ({{.*}}) {
; CHECK-NEXT:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:     VirtualBaseClass {
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       BaseType: A ({{.*}})
; CHECK-NEXT:       VBPtrType: const int* ({{.*}})
; CHECK-NEXT:       VBPtrOffset: 0x0
; CHECK-NEXT:       VBTableIndex: 0x1
; CHECK-NEXT:     }
; CHECK:        }

; Then C's field list.
; CHECK:        FieldList ({{.*}}) {
; CHECK-NEXT:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:     VirtualBaseClass {
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       BaseType: A ({{.*}})
; CHECK-NEXT:       VBPtrType: const int* ({{.*}})
; CHECK-NEXT:       VBPtrOffset: 0x0
; CHECK-NEXT:       VBTableIndex: 0x1
; CHECK-NEXT:     }
; CHECK:        }

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.23918"

%struct.D = type { i32 (...)**, %struct.B.base, %struct.C.base, i32, [4 x i8], %struct.A }
%struct.B.base = type { i32*, i32 }
%struct.C.base = type { i32*, i32 }
%struct.A = type { i32 }
%struct.B = type { i32*, i32, [4 x i8], %struct.A }
%struct.C = type { i32*, i32, [4 x i8], %struct.A }

$"\01??0D@@QEAA@XZ" = comdat any

$"\01??0B@@QEAA@XZ" = comdat any

$"\01??0C@@QEAA@XZ" = comdat any

$"\01??_8D@@7BB@@@" = comdat any

$"\01??_8D@@7BC@@@" = comdat any

$"\01??_7D@@6B@" = comdat any

$"\01??_8B@@7B@" = comdat any

$"\01??_8C@@7B@" = comdat any

@"\01?d@@3UD@@A" = global %struct.D zeroinitializer, align 8, !dbg !4
@"\01??_8D@@7BB@@@" = linkonce_odr unnamed_addr constant [2 x i32] [i32 0, i32 40], comdat
@"\01??_8D@@7BC@@@" = linkonce_odr unnamed_addr constant [2 x i32] [i32 0, i32 24], comdat
@"\01??_7D@@6B@" = linkonce_odr unnamed_addr constant [1 x i8*] [i8* bitcast (void (%struct.D*)* @"\01?f@D@@UEAAXXZ" to i8*)], comdat
@"\01??_8B@@7B@" = linkonce_odr unnamed_addr constant [2 x i32] [i32 0, i32 16], comdat
@"\01??_8C@@7B@" = linkonce_odr unnamed_addr constant [2 x i32] [i32 0, i32 16], comdat
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_t.cpp, i8* null }]

; Function Attrs: uwtable
define internal void @"\01??__Ed@@YAXXZ"() #0 !dbg !37 {
entry:
  %call = call %struct.D* @"\01??0D@@QEAA@XZ"(%struct.D* @"\01?d@@3UD@@A", i32 1) #4, !dbg !40
  ret void, !dbg !40
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr %struct.D* @"\01??0D@@QEAA@XZ"(%struct.D* returned %this, i32 %is_most_derived) unnamed_addr #1 comdat align 2 !dbg !41 {
entry:
  %retval = alloca %struct.D*, align 8
  %is_most_derived.addr = alloca i32, align 4
  %this.addr = alloca %struct.D*, align 8
  store i32 %is_most_derived, i32* %is_most_derived.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %is_most_derived.addr, metadata !43, metadata !44), !dbg !45
  store %struct.D* %this, %struct.D** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.D** %this.addr, metadata !46, metadata !44), !dbg !45
  %this1 = load %struct.D*, %struct.D** %this.addr, align 8
  store %struct.D* %this1, %struct.D** %retval, align 8
  %is_most_derived2 = load i32, i32* %is_most_derived.addr, align 4
  %is_complete_object = icmp ne i32 %is_most_derived2, 0, !dbg !48
  br i1 %is_complete_object, label %ctor.init_vbases, label %ctor.skip_vbases, !dbg !48

ctor.init_vbases:                                 ; preds = %entry
  %this.int8 = bitcast %struct.D* %this1 to i8*, !dbg !48
  %0 = getelementptr inbounds i8, i8* %this.int8, i64 8, !dbg !48
  %vbptr.D = bitcast i8* %0 to i32**, !dbg !48
  store i32* getelementptr inbounds ([2 x i32], [2 x i32]* @"\01??_8D@@7BB@@@", i32 0, i32 0), i32** %vbptr.D, align 8, !dbg !48
  %1 = getelementptr inbounds i8, i8* %this.int8, i64 24, !dbg !48
  %vbptr.C = bitcast i8* %1 to i32**, !dbg !48
  store i32* getelementptr inbounds ([2 x i32], [2 x i32]* @"\01??_8D@@7BC@@@", i32 0, i32 0), i32** %vbptr.C, align 8, !dbg !48
  %2 = bitcast %struct.D* %this1 to i8*, !dbg !48
  %3 = getelementptr inbounds i8, i8* %2, i64 48, !dbg !48
  %4 = bitcast i8* %3 to %struct.A*, !dbg !48
  br label %ctor.skip_vbases, !dbg !48

ctor.skip_vbases:                                 ; preds = %ctor.init_vbases, %entry
  %5 = bitcast %struct.D* %this1 to i8*, !dbg !48
  %6 = getelementptr inbounds i8, i8* %5, i64 8, !dbg !48
  %7 = bitcast i8* %6 to %struct.B*, !dbg !48
  %call = call %struct.B* @"\01??0B@@QEAA@XZ"(%struct.B* %7, i32 0) #4, !dbg !48
  %8 = bitcast %struct.D* %this1 to i8*, !dbg !48
  %9 = getelementptr inbounds i8, i8* %8, i64 24, !dbg !48
  %10 = bitcast i8* %9 to %struct.C*, !dbg !48
  %call3 = call %struct.C* @"\01??0C@@QEAA@XZ"(%struct.C* %10, i32 0) #4, !dbg !48
  %11 = bitcast %struct.D* %this1 to i32 (...)***, !dbg !48
  store i32 (...)** bitcast ([1 x i8*]* @"\01??_7D@@6B@" to i32 (...)**), i32 (...)*** %11, align 8, !dbg !48
  %12 = load %struct.D*, %struct.D** %retval, align 8, !dbg !48
  ret %struct.D* %12, !dbg !48
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr %struct.B* @"\01??0B@@QEAA@XZ"(%struct.B* returned %this, i32 %is_most_derived) unnamed_addr #1 comdat align 2 !dbg !49 {
entry:
  %retval = alloca %struct.B*, align 8
  %is_most_derived.addr = alloca i32, align 4
  %this.addr = alloca %struct.B*, align 8
  store i32 %is_most_derived, i32* %is_most_derived.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %is_most_derived.addr, metadata !54, metadata !44), !dbg !55
  store %struct.B* %this, %struct.B** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.B** %this.addr, metadata !56, metadata !44), !dbg !55
  %this1 = load %struct.B*, %struct.B** %this.addr, align 8
  store %struct.B* %this1, %struct.B** %retval, align 8
  %is_most_derived2 = load i32, i32* %is_most_derived.addr, align 4
  %is_complete_object = icmp ne i32 %is_most_derived2, 0, !dbg !58
  br i1 %is_complete_object, label %ctor.init_vbases, label %ctor.skip_vbases, !dbg !58

ctor.init_vbases:                                 ; preds = %entry
  %this.int8 = bitcast %struct.B* %this1 to i8*, !dbg !58
  %0 = getelementptr inbounds i8, i8* %this.int8, i64 0, !dbg !58
  %vbptr.B = bitcast i8* %0 to i32**, !dbg !58
  store i32* getelementptr inbounds ([2 x i32], [2 x i32]* @"\01??_8B@@7B@", i32 0, i32 0), i32** %vbptr.B, align 8, !dbg !58
  %1 = bitcast %struct.B* %this1 to i8*, !dbg !58
  %2 = getelementptr inbounds i8, i8* %1, i64 16, !dbg !58
  %3 = bitcast i8* %2 to %struct.A*, !dbg !58
  br label %ctor.skip_vbases, !dbg !58

ctor.skip_vbases:                                 ; preds = %ctor.init_vbases, %entry
  %4 = load %struct.B*, %struct.B** %retval, align 8, !dbg !58
  ret %struct.B* %4, !dbg !58
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr %struct.C* @"\01??0C@@QEAA@XZ"(%struct.C* returned %this, i32 %is_most_derived) unnamed_addr #1 comdat align 2 !dbg !59 {
entry:
  %retval = alloca %struct.C*, align 8
  %is_most_derived.addr = alloca i32, align 4
  %this.addr = alloca %struct.C*, align 8
  store i32 %is_most_derived, i32* %is_most_derived.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %is_most_derived.addr, metadata !64, metadata !44), !dbg !65
  store %struct.C* %this, %struct.C** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.C** %this.addr, metadata !66, metadata !44), !dbg !65
  %this1 = load %struct.C*, %struct.C** %this.addr, align 8
  store %struct.C* %this1, %struct.C** %retval, align 8
  %is_most_derived2 = load i32, i32* %is_most_derived.addr, align 4
  %is_complete_object = icmp ne i32 %is_most_derived2, 0, !dbg !68
  br i1 %is_complete_object, label %ctor.init_vbases, label %ctor.skip_vbases, !dbg !68

ctor.init_vbases:                                 ; preds = %entry
  %this.int8 = bitcast %struct.C* %this1 to i8*, !dbg !68
  %0 = getelementptr inbounds i8, i8* %this.int8, i64 0, !dbg !68
  %vbptr.C = bitcast i8* %0 to i32**, !dbg !68
  store i32* getelementptr inbounds ([2 x i32], [2 x i32]* @"\01??_8C@@7B@", i32 0, i32 0), i32** %vbptr.C, align 8, !dbg !68
  %1 = bitcast %struct.C* %this1 to i8*, !dbg !68
  %2 = getelementptr inbounds i8, i8* %1, i64 16, !dbg !68
  %3 = bitcast i8* %2 to %struct.A*, !dbg !68
  br label %ctor.skip_vbases, !dbg !68

ctor.skip_vbases:                                 ; preds = %ctor.init_vbases, %entry
  %4 = load %struct.C*, %struct.C** %retval, align 8, !dbg !68
  ret %struct.C* %4, !dbg !68
}

declare void @"\01?f@D@@UEAAXXZ"(%struct.D*) unnamed_addr #3

; Function Attrs: uwtable
define internal void @_GLOBAL__sub_I_t.cpp() #0 !dbg !69 {
entry:
  call void @"\01??__Ed@@YAXXZ"(), !dbg !71
  ret void
}

attributes #0 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { inlinehint nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!33, !34, !35}
!llvm.ident = !{!36}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!2 = !{}
!3 = !{!4}
!4 = distinct !DIGlobalVariable(name: "d", linkageName: "\01?d@@3UD@@A", scope: !0, file: !1, line: 9, type: !5, isLocal: false, isDefinition: true)
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "D", file: !1, line: 4, size: 448, align: 64, elements: !6, vtableHolder: !5, identifier: ".?AUD@@")
!6 = !{!7, !21, !27, !28, !29}
!7 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !5, baseType: !8, offset: 64)
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "B", file: !1, line: 2, size: 192, align: 64, elements: !9, vtableHolder: !8, identifier: ".?AUB@@")
!9 = !{!10, !15, !20}
!10 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !8, baseType: !11, offset: 4, flags: DIFlagVirtual)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !1, line: 1, size: 32, align: 32, elements: !12, identifier: ".?AUA@@")
!12 = !{!13}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !11, file: !1, line: 1, baseType: !14, size: 32, align: 32)
!14 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$B", scope: !1, file: !1, baseType: !16, size: 64, flags: DIFlagArtificial)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !17, size: 64)
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "__vtbl_ptr_type", baseType: !18, size: 64)
!18 = !DISubroutineType(types: !19)
!19 = !{!14}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !8, file: !1, line: 2, baseType: !14, size: 32, align: 32, offset: 64)
!21 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !5, baseType: !22, offset: 192)
!22 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "C", file: !1, line: 3, size: 192, align: 64, elements: !23, vtableHolder: !22, identifier: ".?AUC@@")
!23 = !{!24, !25, !26}
!24 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !22, baseType: !11, offset: 4, flags: DIFlagVirtual)
!25 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$C", scope: !1, file: !1, baseType: !16, size: 64, flags: DIFlagArtificial)
!26 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !22, file: !1, line: 3, baseType: !14, size: 32, align: 32, offset: 64)
!27 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$D", scope: !1, file: !1, baseType: !16, size: 64, flags: DIFlagArtificial)
!28 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !5, file: !1, line: 6, baseType: !14, size: 32, align: 32, offset: 320)
!29 = !DISubprogram(name: "f", linkageName: "\01?f@D@@UEAAXXZ", scope: !5, file: !1, line: 5, type: !30, isLocal: false, isDefinition: false, scopeLine: 5, containingType: !5, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 0, flags: DIFlagPrototyped | DIFlagIntroducedVirtual, isOptimized: false)
!30 = !DISubroutineType(types: !31)
!31 = !{null, !32}
!32 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!33 = !{i32 2, !"CodeView", i32 1}
!34 = !{i32 2, !"Debug Info Version", i32 3}
!35 = !{i32 1, !"PIC Level", i32 2}
!36 = !{!"clang version 3.9.0 "}
!37 = distinct !DISubprogram(name: "??__Ed@@YAXXZ", scope: !1, file: !1, line: 9, type: !38, isLocal: true, isDefinition: true, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!38 = !DISubroutineType(types: !39)
!39 = !{null}
!40 = !DILocation(line: 9, column: 3, scope: !37)
!41 = distinct !DISubprogram(name: "D", linkageName: "\01??0D@@QEAA@XZ", scope: !5, file: !1, line: 4, type: !30, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !42, variables: !2)
!42 = !DISubprogram(name: "D", scope: !5, type: !30, isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!43 = !DILocalVariable(name: "is_most_derived", arg: 2, scope: !41, type: !14, flags: DIFlagArtificial)
!44 = !DIExpression()
!45 = !DILocation(line: 0, scope: !41)
!46 = !DILocalVariable(name: "this", arg: 1, scope: !41, type: !47, flags: DIFlagArtificial | DIFlagObjectPointer)
!47 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64, align: 64)
!48 = !DILocation(line: 4, column: 8, scope: !41)
!49 = distinct !DISubprogram(name: "B", linkageName: "\01??0B@@QEAA@XZ", scope: !8, file: !1, line: 2, type: !50, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !53, variables: !2)
!50 = !DISubroutineType(types: !51)
!51 = !{null, !52}
!52 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!53 = !DISubprogram(name: "B", scope: !8, type: !50, isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!54 = !DILocalVariable(name: "is_most_derived", arg: 2, scope: !49, type: !14, flags: DIFlagArtificial)
!55 = !DILocation(line: 0, scope: !49)
!56 = !DILocalVariable(name: "this", arg: 1, scope: !49, type: !57, flags: DIFlagArtificial | DIFlagObjectPointer)
!57 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64, align: 64)
!58 = !DILocation(line: 2, column: 8, scope: !49)
!59 = distinct !DISubprogram(name: "C", linkageName: "\01??0C@@QEAA@XZ", scope: !22, file: !1, line: 3, type: !60, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !63, variables: !2)
!60 = !DISubroutineType(types: !61)
!61 = !{null, !62}
!62 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!63 = !DISubprogram(name: "C", scope: !22, type: !60, isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!64 = !DILocalVariable(name: "is_most_derived", arg: 2, scope: !59, type: !14, flags: DIFlagArtificial)
!65 = !DILocation(line: 0, scope: !59)
!66 = !DILocalVariable(name: "this", arg: 1, scope: !59, type: !67, flags: DIFlagArtificial | DIFlagObjectPointer)
!67 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 64, align: 64)
!68 = !DILocation(line: 3, column: 8, scope: !59)
!69 = distinct !DISubprogram(linkageName: "_GLOBAL__sub_I_t.cpp", scope: !1, file: !1, type: !70, isLocal: true, isDefinition: true, flags: DIFlagArtificial, isOptimized: false, unit: !0, variables: !2)
!70 = !DISubroutineType(types: !2)
!71 = !DILocation(line: 0, scope: !69)
