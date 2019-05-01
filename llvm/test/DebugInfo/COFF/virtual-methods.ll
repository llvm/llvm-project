; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s

; C++ source used to generate IR:
; $ cat t.cpp
; struct A {
;   virtual int f();
; };
; struct B {
;   virtual int g() = 0;
; };
; struct C : A, B {
;   int c = 42;
;   virtual int g();
; };
; int C::g() { return c; }
; struct D : virtual B {
;   int d = 13;
;   virtual int g();
; };
; int D::g() { return d; }
; $ clang t.cpp -S -emit-llvm -fstandalone-debug -g -gcodeview -o t.ll

; A::f
; CHECK:      MemberFunction ({{.*}}) {
; CHECK-NEXT:   TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK-NEXT:   ReturnType: int (0x74)
; CHECK-NEXT:   ClassType: A ({{.*}})
; CHECK-NEXT:   ThisType: A* const ({{.*}})
; CHECK-NEXT:   CallingConvention: NearC (0x0)
; CHECK-NEXT:   FunctionOptions [ (0x0)
; CHECK-NEXT:   ]
; CHECK-NEXT:   NumParameters: 0
; CHECK-NEXT:   ArgListType: () ({{.*}})
; CHECK-NEXT:   ThisAdjustment: 0
; CHECK-NEXT: }

; A::g
; CHECK:      MemberFunction ({{.*}}) {
; CHECK-NEXT:   TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK-NEXT:   ReturnType: int (0x74)
; CHECK-NEXT:   ClassType: B ({{.*}})
; CHECK-NEXT:   ThisType: B* const ({{.*}})
; CHECK-NEXT:   CallingConvention: NearC (0x0)
; CHECK-NEXT:   FunctionOptions [ (0x0)
; CHECK-NEXT:   ]
; CHECK-NEXT:   NumParameters: 0
; CHECK-NEXT:   ArgListType: () ({{.*}})
; CHECK-NEXT:   ThisAdjustment: 0
; CHECK-NEXT: }

; C::g
; CHECK:      MemberFunction ([[C_g:.*]]) {
; CHECK-NEXT:   TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK-NEXT:   ReturnType: int (0x74)
; CHECK-NEXT:   ClassType: C ({{.*}})
; CHECK-NEXT:   ThisType: C* const ({{.*}})
; CHECK-NEXT:   CallingConvention: NearC (0x0)
; CHECK-NEXT:   FunctionOptions [ (0x0)
; CHECK-NEXT:   ]
; CHECK-NEXT:   NumParameters: 0
; CHECK-NEXT:   ArgListType: () ({{.*}})
; CHECK-NEXT:   ThisAdjustment: 8
; CHECK-NEXT: }

; CHECK:      FieldList ({{.*}}) {
; CHECK:        OneMethod {
; CHECK-NEXT:     TypeLeafKind: LF_ONEMETHOD (0x1511)
; CHECK-NEXT:     AccessSpecifier: Public (0x3)
; CHECK-NEXT:     MethodKind: Virtual (0x1)
; CHECK-NEXT:     Type: int C::() ([[C_g]])
; CHECK-NEXT:     Name: g
; CHECK-NEXT:   }
; CHECK-NEXT: }

; D::g
; CHECK:      MemberFunction ([[D_g:.*]]) {
; CHECK-NEXT:   TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK-NEXT:   ReturnType: int (0x74)
; CHECK-NEXT:   ClassType: D ({{.*}})
; CHECK-NEXT:   ThisType: D* const ({{.*}})
; CHECK-NEXT:   CallingConvention: NearC (0x0)
; CHECK-NEXT:   FunctionOptions [ (0x0)
; CHECK-NEXT:   ]
; CHECK-NEXT:   NumParameters: 0
; CHECK-NEXT:   ArgListType: () ({{.*}})
; CHECK-NEXT:   ThisAdjustment: 16
; CHECK-NEXT: }

; CHECK:      FieldList ({{.*}}) {
; CHECK:        OneMethod {
; CHECK-NEXT:     TypeLeafKind: LF_ONEMETHOD (0x1511)
; CHECK-NEXT:     AccessSpecifier: Public (0x3)
; CHECK-NEXT:     MethodKind: Virtual (0x1)
; CHECK-NEXT:     Type: int D::() ([[D_g]])
; CHECK-NEXT:     Name: g
; CHECK-NEXT:   }
; CHECK-NEXT: }

; Need to skip constructor IDs...
; CHECK: MemberFuncId ({{.*}}) {
; CHECK:   TypeLeafKind: LF_MFUNC_ID (0x1602)
; CHECK:   ClassType: A
; CHECK:   FunctionType: void A::()
; CHECK:   Name: A
; CHECK: }
; CHECK: MemberFuncId ({{.*}}) {
; CHECK:   TypeLeafKind: LF_MFUNC_ID (0x1602)
; CHECK:   ClassType: B
; CHECK:   FunctionType: void B::()
; CHECK:   Name: B
; CHECK: }
; CHECK: MemberFuncId ({{.*}}) {
; CHECK:   TypeLeafKind: LF_MFUNC_ID (0x1602)
; CHECK:   ClassType: C
; CHECK:   FunctionType: void C::()
; CHECK:   Name: C
; CHECK: }
; CHECK: MemberFuncId ({{.*}}) {
; CHECK:   TypeLeafKind: LF_MFUNC_ID (0x1602)
; CHECK:   ClassType: D
; CHECK:   FunctionType: void D::()
; CHECK:   Name: D
; CHECK: }

; CHECK:      MemberFuncId ({{.*}}) {
; CHECK-NEXT:   TypeLeafKind: LF_MFUNC_ID (0x1602)
; CHECK-NEXT:   ClassType: C ({{.*}})
; CHECK-NEXT:   FunctionType: int C::() ([[C_g]])
; CHECK-NEXT:   Name: g
; CHECK-NEXT: }

; CHECK:      MemberFuncId ({{.*}}) {
; CHECK-NEXT:   TypeLeafKind: LF_MFUNC_ID (0x1602)
; CHECK-NEXT:   ClassType: D ({{.*}})
; CHECK-NEXT:   FunctionType: int D::() ([[D_g]])
; CHECK-NEXT:   Name: g
; CHECK-NEXT: }

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.23918"

%struct.A = type { i32 (...)** }
%struct.B = type { i32 (...)** }
%struct.C = type { %struct.A, %struct.B, i32 }
%struct.D = type { i32*, i32, %struct.B }

$"\01??0A@@QEAA@XZ" = comdat any

$"\01??0B@@QEAA@XZ" = comdat any

$"\01??0C@@QEAA@XZ" = comdat any

$"\01??0D@@QEAA@XZ" = comdat any

$"\01?g@C@@UEAAHXZ" = comdat any

$"\01?g@D@@UEAAHXZ" = comdat any

$"\01??_7A@@6B@" = comdat any

$"\01??_7B@@6B@" = comdat any

$"\01??_7C@@6BA@@@" = comdat any

$"\01??_7C@@6BB@@@" = comdat any

$"\01??_8D@@7B@" = comdat any

$"\01??_7D@@6B@" = comdat any

@"\01??_7A@@6B@" = linkonce_odr unnamed_addr constant [1 x i8*] [i8* bitcast (i32 (%struct.A*)* @"\01?f@A@@UEAAHXZ" to i8*)], comdat
@"\01??_7B@@6B@" = linkonce_odr unnamed_addr constant [1 x i8*] [i8* bitcast (i32 (%struct.B*)* @"\01?g@B@@UEAAHXZ" to i8*)], comdat
@"\01??_7C@@6BA@@@" = linkonce_odr unnamed_addr constant [1 x i8*] [i8* bitcast (i32 (%struct.A*)* @"\01?f@A@@UEAAHXZ" to i8*)], comdat
@"\01??_7C@@6BB@@@" = linkonce_odr unnamed_addr constant [1 x i8*] [i8* bitcast (i32 (i8*)* @"\01?g@C@@UEAAHXZ" to i8*)], comdat
@"\01??_8D@@7B@" = linkonce_odr unnamed_addr constant [2 x i32] [i32 0, i32 16], comdat
@"\01??_7D@@6B@" = linkonce_odr unnamed_addr constant [1 x i8*] [i8* bitcast (i32 (i8*)* @"\01?g@D@@UEAAHXZ" to i8*)], comdat

; Function Attrs: uwtable
define void @"\01?usetypes@@YAXXZ"() #0 !dbg !7 {
entry:
  %a = alloca %struct.A, align 8
  %b = alloca %struct.B, align 8
  %c = alloca %struct.C, align 8
  %d = alloca %struct.D, align 8
  call void @llvm.dbg.declare(metadata %struct.A* %a, metadata !10, metadata !23), !dbg !24
  %call = call %struct.A* @"\01??0A@@QEAA@XZ"(%struct.A* %a) #5, !dbg !24
  call void @llvm.dbg.declare(metadata %struct.B* %b, metadata !25, metadata !23), !dbg !33
  %call1 = call %struct.B* @"\01??0B@@QEAA@XZ"(%struct.B* %b) #5, !dbg !33
  call void @llvm.dbg.declare(metadata %struct.C* %c, metadata !34, metadata !23), !dbg !44
  %call2 = call %struct.C* @"\01??0C@@QEAA@XZ"(%struct.C* %c) #5, !dbg !44
  call void @llvm.dbg.declare(metadata %struct.D* %d, metadata !45, metadata !23), !dbg !55
  %call3 = call %struct.D* @"\01??0D@@QEAA@XZ"(%struct.D* %d, i32 1) #5, !dbg !55
  %0 = bitcast %struct.C* %c to i8*, !dbg !56
  %1 = getelementptr i8, i8* %0, i64 8, !dbg !56
  %call4 = call i32 @"\01?g@C@@UEAAHXZ"(i8* %1), !dbg !56
  ret void, !dbg !57
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr %struct.A* @"\01??0A@@QEAA@XZ"(%struct.A* returned %this) unnamed_addr #2 comdat align 2 !dbg !58 {
entry:
  %this.addr = alloca %struct.A*, align 8
  store %struct.A* %this, %struct.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.A** %this.addr, metadata !62, metadata !23), !dbg !64
  %this1 = load %struct.A*, %struct.A** %this.addr, align 8
  %0 = bitcast %struct.A* %this1 to i32 (...)***, !dbg !65
  store i32 (...)** bitcast ([1 x i8*]* @"\01??_7A@@6B@" to i32 (...)**), i32 (...)*** %0, align 8, !dbg !65
  ret %struct.A* %this1, !dbg !65
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr %struct.B* @"\01??0B@@QEAA@XZ"(%struct.B* returned %this) unnamed_addr #2 comdat align 2 !dbg !66 {
entry:
  %this.addr = alloca %struct.B*, align 8
  store %struct.B* %this, %struct.B** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.B** %this.addr, metadata !70, metadata !23), !dbg !72
  %this1 = load %struct.B*, %struct.B** %this.addr, align 8
  %0 = bitcast %struct.B* %this1 to i32 (...)***, !dbg !73
  store i32 (...)** bitcast ([1 x i8*]* @"\01??_7B@@6B@" to i32 (...)**), i32 (...)*** %0, align 8, !dbg !73
  ret %struct.B* %this1, !dbg !73
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr %struct.C* @"\01??0C@@QEAA@XZ"(%struct.C* returned %this) unnamed_addr #2 comdat align 2 !dbg !74 {
entry:
  %this.addr = alloca %struct.C*, align 8
  store %struct.C* %this, %struct.C** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.C** %this.addr, metadata !78, metadata !23), !dbg !80
  %this1 = load %struct.C*, %struct.C** %this.addr, align 8
  %0 = bitcast %struct.C* %this1 to %struct.A*, !dbg !81
  %call = call %struct.A* @"\01??0A@@QEAA@XZ"(%struct.A* %0) #5, !dbg !81
  %1 = bitcast %struct.C* %this1 to i8*, !dbg !81
  %2 = getelementptr inbounds i8, i8* %1, i64 8, !dbg !81
  %3 = bitcast i8* %2 to %struct.B*, !dbg !81
  %call2 = call %struct.B* @"\01??0B@@QEAA@XZ"(%struct.B* %3) #5, !dbg !81
  %4 = bitcast %struct.C* %this1 to i32 (...)***, !dbg !81
  store i32 (...)** bitcast ([1 x i8*]* @"\01??_7C@@6BA@@@" to i32 (...)**), i32 (...)*** %4, align 8, !dbg !81
  %5 = bitcast %struct.C* %this1 to i8*, !dbg !81
  %add.ptr = getelementptr inbounds i8, i8* %5, i64 8, !dbg !81
  %6 = bitcast i8* %add.ptr to i32 (...)***, !dbg !81
  store i32 (...)** bitcast ([1 x i8*]* @"\01??_7C@@6BB@@@" to i32 (...)**), i32 (...)*** %6, align 8, !dbg !81
  %c = getelementptr inbounds %struct.C, %struct.C* %this1, i32 0, i32 2, !dbg !82
  store i32 42, i32* %c, align 8, !dbg !82
  ret %struct.C* %this1, !dbg !81
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr %struct.D* @"\01??0D@@QEAA@XZ"(%struct.D* returned %this, i32 %is_most_derived) unnamed_addr #2 comdat align 2 !dbg !83 {
entry:
  %retval = alloca %struct.D*, align 8
  %is_most_derived.addr = alloca i32, align 4
  %this.addr = alloca %struct.D*, align 8
  store i32 %is_most_derived, i32* %is_most_derived.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %is_most_derived.addr, metadata !87, metadata !23), !dbg !88
  store %struct.D* %this, %struct.D** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.D** %this.addr, metadata !89, metadata !23), !dbg !88
  %this1 = load %struct.D*, %struct.D** %this.addr, align 8
  store %struct.D* %this1, %struct.D** %retval, align 8
  %is_most_derived2 = load i32, i32* %is_most_derived.addr, align 4
  %is_complete_object = icmp ne i32 %is_most_derived2, 0, !dbg !91
  br i1 %is_complete_object, label %ctor.init_vbases, label %ctor.skip_vbases, !dbg !91

ctor.init_vbases:                                 ; preds = %entry
  %this.int8 = bitcast %struct.D* %this1 to i8*, !dbg !91
  %0 = getelementptr inbounds i8, i8* %this.int8, i64 0, !dbg !91
  %vbptr.D = bitcast i8* %0 to i32**, !dbg !91
  store i32* getelementptr inbounds ([2 x i32], [2 x i32]* @"\01??_8D@@7B@", i32 0, i32 0), i32** %vbptr.D, align 8, !dbg !91
  %1 = bitcast %struct.D* %this1 to i8*, !dbg !91
  %2 = getelementptr inbounds i8, i8* %1, i64 16, !dbg !91
  %3 = bitcast i8* %2 to %struct.B*, !dbg !91
  %call = call %struct.B* @"\01??0B@@QEAA@XZ"(%struct.B* %3) #5, !dbg !91
  br label %ctor.skip_vbases, !dbg !91

ctor.skip_vbases:                                 ; preds = %ctor.init_vbases, %entry
  %4 = bitcast %struct.D* %this1 to i8*, !dbg !91
  %vbptr = getelementptr inbounds i8, i8* %4, i64 0, !dbg !91
  %5 = bitcast i8* %vbptr to i32**, !dbg !91
  %vbtable = load i32*, i32** %5, align 8, !dbg !91
  %6 = getelementptr inbounds i32, i32* %vbtable, i32 1, !dbg !91
  %vbase_offs = load i32, i32* %6, align 4, !dbg !91
  %7 = sext i32 %vbase_offs to i64, !dbg !91
  %8 = add nsw i64 0, %7, !dbg !91
  %9 = bitcast %struct.D* %this1 to i8*, !dbg !91
  %add.ptr = getelementptr inbounds i8, i8* %9, i64 %8, !dbg !91
  %10 = bitcast i8* %add.ptr to i32 (...)***, !dbg !91
  store i32 (...)** bitcast ([1 x i8*]* @"\01??_7D@@6B@" to i32 (...)**), i32 (...)*** %10, align 8, !dbg !91
  %d = getelementptr inbounds %struct.D, %struct.D* %this1, i32 0, i32 1, !dbg !92
  store i32 13, i32* %d, align 8, !dbg !92
  %11 = load %struct.D*, %struct.D** %retval, align 8, !dbg !91
  ret %struct.D* %11, !dbg !91
}

; Function Attrs: nounwind uwtable
define linkonce_odr i32 @"\01?g@C@@UEAAHXZ"(i8*) unnamed_addr #3 comdat align 2 !dbg !93 {
entry:
  %this.addr = alloca %struct.C*, align 8
  %1 = getelementptr inbounds i8, i8* %0, i32 -8
  %this = bitcast i8* %1 to %struct.C*
  store %struct.C* %this, %struct.C** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.C** %this.addr, metadata !94, metadata !23), !dbg !95
  %this1 = load %struct.C*, %struct.C** %this.addr, align 8
  %c = getelementptr inbounds %struct.C, %struct.C* %this1, i32 0, i32 2, !dbg !96
  %2 = load i32, i32* %c, align 8, !dbg !96
  ret i32 %2, !dbg !97
}

declare i32 @"\01?f@A@@UEAAHXZ"(%struct.A*) unnamed_addr #4

declare i32 @"\01?g@B@@UEAAHXZ"(%struct.B*) unnamed_addr #4

; Function Attrs: nounwind uwtable
define linkonce_odr i32 @"\01?g@D@@UEAAHXZ"(i8*) unnamed_addr #3 comdat align 2 !dbg !98 {
entry:
  %this.addr = alloca %struct.D*, align 8
  %1 = getelementptr inbounds i8, i8* %0, i32 -16
  %this = bitcast i8* %1 to %struct.D*
  store %struct.D* %this, %struct.D** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.D** %this.addr, metadata !99, metadata !23), !dbg !100
  %this1 = load %struct.D*, %struct.D** %this.addr, align 8
  %d = getelementptr inbounds %struct.D, %struct.D* %this1, i32 0, i32 1, !dbg !101
  %2 = load i32, i32* %d, align 8, !dbg !101
  ret i32 %2, !dbg !102
}

attributes #0 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { inlinehint nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 3.9.0 "}
!7 = distinct !DISubprogram(name: "usetypes", linkageName: "\01?usetypes@@YAXXZ", scope: !1, file: !1, line: 15, type: !8, isLocal: false, isDefinition: true, scopeLine: 15, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocalVariable(name: "a", scope: !7, file: !1, line: 16, type: !11)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !1, line: 1, size: 64, align: 64, elements: !12, vtableHolder: !11, identifier: ".?AUA@@")
!12 = !{!13, !19}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$A", scope: !1, file: !1, baseType: !14, size: 64, flags: DIFlagArtificial)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "__vtbl_ptr_type", baseType: !16, size: 64)
!16 = !DISubroutineType(types: !17)
!17 = !{!18}
!18 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!19 = !DISubprogram(name: "f", linkageName: "\01?f@A@@UEAAHXZ", scope: !11, file: !1, line: 2, type: !20, isLocal: false, isDefinition: false, scopeLine: 2, containingType: !11, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 0, flags: DIFlagPrototyped | DIFlagIntroducedVirtual, isOptimized: false)
!20 = !DISubroutineType(types: !21)
!21 = !{!18, !22}
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!23 = !DIExpression()
!24 = !DILocation(line: 16, column: 5, scope: !7)
!25 = !DILocalVariable(name: "b", scope: !7, file: !1, line: 17, type: !26)
!26 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "B", file: !1, line: 4, size: 64, align: 64, elements: !27, vtableHolder: !26, identifier: ".?AUB@@")
!27 = !{!28, !29}
!28 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$B", scope: !1, file: !1, baseType: !14, size: 64, flags: DIFlagArtificial)
!29 = !DISubprogram(name: "g", linkageName: "\01?g@B@@UEAAHXZ", scope: !26, file: !1, line: 5, type: !30, isLocal: false, isDefinition: false, scopeLine: 5, containingType: !26, virtuality: DW_VIRTUALITY_pure_virtual, virtualIndex: 0, flags: DIFlagPrototyped | DIFlagIntroducedVirtual, isOptimized: false)
!30 = !DISubroutineType(types: !31)
!31 = !{!18, !32}
!32 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !26, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!33 = !DILocation(line: 17, column: 5, scope: !7)
!34 = !DILocalVariable(name: "c", scope: !7, file: !1, line: 18, type: !35)
!35 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "C", file: !1, line: 7, size: 192, align: 64, elements: !36, vtableHolder: !11, identifier: ".?AUC@@")
!36 = !{!37, !38, !39, !40}
!37 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !35, baseType: !11)
!38 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !35, baseType: !26, offset: 64)
!39 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !35, file: !1, line: 8, baseType: !18, size: 32, align: 32, offset: 128)
!40 = !DISubprogram(name: "g", linkageName: "\01?g@C@@UEAAHXZ", scope: !35, file: !1, line: 9, type: !41, isLocal: false, isDefinition: false, scopeLine: 9, containingType: !35, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 0, thisAdjustment: 8, flags: DIFlagPrototyped, isOptimized: false)
!41 = !DISubroutineType(types: !42)
!42 = !{!18, !43}
!43 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !35, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!44 = !DILocation(line: 18, column: 5, scope: !7)
!45 = !DILocalVariable(name: "d", scope: !7, file: !1, line: 19, type: !46)
!46 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "D", file: !1, line: 11, size: 192, align: 64, elements: !47, vtableHolder: !46, identifier: ".?AUD@@")
!47 = !{!48, !49, !50, !51}
!48 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !46, baseType: !26, offset: 4, flags: DIFlagVirtual)
!49 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$D", scope: !1, file: !1, baseType: !14, size: 64, flags: DIFlagArtificial)
!50 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !46, file: !1, line: 12, baseType: !18, size: 32, align: 32, offset: 64)
!51 = !DISubprogram(name: "g", linkageName: "\01?g@D@@UEAAHXZ", scope: !46, file: !1, line: 13, type: !52, isLocal: false, isDefinition: false, scopeLine: 13, containingType: !46, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 0, thisAdjustment: 16, flags: DIFlagPrototyped, isOptimized: false)
!52 = !DISubroutineType(types: !53)
!53 = !{!18, !54}
!54 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !46, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!55 = !DILocation(line: 19, column: 5, scope: !7)
!56 = !DILocation(line: 20, column: 5, scope: !7)
!57 = !DILocation(line: 21, column: 1, scope: !7)
!58 = distinct !DISubprogram(name: "A", linkageName: "\01??0A@@QEAA@XZ", scope: !11, file: !1, line: 1, type: !59, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !61, retainedNodes: !2)
!59 = !DISubroutineType(types: !60)
!60 = !{null, !22}
!61 = !DISubprogram(name: "A", scope: !11, type: !59, isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!62 = !DILocalVariable(name: "this", arg: 1, scope: !58, type: !63, flags: DIFlagArtificial | DIFlagObjectPointer)
!63 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64, align: 64)
!64 = !DILocation(line: 0, scope: !58)
!65 = !DILocation(line: 1, column: 8, scope: !58)
!66 = distinct !DISubprogram(name: "B", linkageName: "\01??0B@@QEAA@XZ", scope: !26, file: !1, line: 4, type: !67, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !69, retainedNodes: !2)
!67 = !DISubroutineType(types: !68)
!68 = !{null, !32}
!69 = !DISubprogram(name: "B", scope: !26, type: !67, isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!70 = !DILocalVariable(name: "this", arg: 1, scope: !66, type: !71, flags: DIFlagArtificial | DIFlagObjectPointer)
!71 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !26, size: 64, align: 64)
!72 = !DILocation(line: 0, scope: !66)
!73 = !DILocation(line: 4, column: 8, scope: !66)
!74 = distinct !DISubprogram(name: "C", linkageName: "\01??0C@@QEAA@XZ", scope: !35, file: !1, line: 7, type: !75, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !77, retainedNodes: !2)
!75 = !DISubroutineType(types: !76)
!76 = !{null, !43}
!77 = !DISubprogram(name: "C", scope: !35, type: !75, isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!78 = !DILocalVariable(name: "this", arg: 1, scope: !74, type: !79, flags: DIFlagArtificial | DIFlagObjectPointer)
!79 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !35, size: 64, align: 64)
!80 = !DILocation(line: 0, scope: !74)
!81 = !DILocation(line: 7, column: 8, scope: !74)
!82 = !DILocation(line: 8, column: 7, scope: !74)
!83 = distinct !DISubprogram(name: "D", linkageName: "\01??0D@@QEAA@XZ", scope: !46, file: !1, line: 11, type: !84, isLocal: false, isDefinition: true, scopeLine: 11, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !86, retainedNodes: !2)
!84 = !DISubroutineType(types: !85)
!85 = !{null, !54}
!86 = !DISubprogram(name: "D", scope: !46, type: !84, isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!87 = !DILocalVariable(name: "is_most_derived", arg: 2, scope: !83, type: !18, flags: DIFlagArtificial)
!88 = !DILocation(line: 0, scope: !83)
!89 = !DILocalVariable(name: "this", arg: 1, scope: !83, type: !90, flags: DIFlagArtificial | DIFlagObjectPointer)
!90 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !46, size: 64, align: 64)
!91 = !DILocation(line: 11, column: 8, scope: !83)
!92 = !DILocation(line: 12, column: 7, scope: !83)
!93 = distinct !DISubprogram(name: "g", linkageName: "\01?g@C@@UEAAHXZ", scope: !35, file: !1, line: 9, type: !41, isLocal: false, isDefinition: true, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !40, retainedNodes: !2)
!94 = !DILocalVariable(name: "this", arg: 1, scope: !93, type: !79, flags: DIFlagArtificial | DIFlagObjectPointer)
!95 = !DILocation(line: 0, scope: !93)
!96 = !DILocation(line: 9, column: 28, scope: !93)
!97 = !DILocation(line: 9, column: 21, scope: !93)
!98 = distinct !DISubprogram(name: "g", linkageName: "\01?g@D@@UEAAHXZ", scope: !46, file: !1, line: 13, type: !52, isLocal: false, isDefinition: true, scopeLine: 13, flags: DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !51, retainedNodes: !2)
!99 = !DILocalVariable(name: "this", arg: 1, scope: !98, type: !90, flags: DIFlagArtificial | DIFlagObjectPointer)
!100 = !DILocation(line: 0, scope: !98)
!101 = !DILocation(line: 13, column: 28, scope: !98)
!102 = !DILocation(line: 13, column: 21, scope: !98)
