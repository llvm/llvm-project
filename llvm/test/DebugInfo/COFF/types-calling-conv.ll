; RUN: llc < %s -filetype=obj -o - | llvm-readobj - -codeview | FileCheck %s

; C++ source to regenerate:
; $ cat t.cpp
; struct A {
;   void thiscallcc();
; };
; void A::thiscallcc() {}
; void cdeclcc() {}
; void __fastcall fastcallcc() {}
; void __stdcall stdcallcc() {}
; void __vectorcall vectorcallcc() {}
; $ clang -g -gcodeview t.cpp -emit-llvm -S -o t.ll -O1

; CHECK: CodeViewTypes [
; CHECK:   Section: .debug$T (5)
; CHECK:   Magic: 0x4
; CHECK:   Struct (0x1000) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 0
; CHECK:     Properties [ (0x80)
; CHECK:       ForwardReference (0x80)
; CHECK:     ]
; CHECK:     FieldList: 0x0
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 0
; CHECK:     Name: A
; CHECK:   }
; CHECK:   Pointer (0x1001) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: A (0x1000)
; CHECK:     PointerAttributes: 0x800A
; CHECK:     PtrType: Near32 (0xA)
; CHECK:     PtrMode: Pointer (0x0)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     SizeOf: 4
; CHECK:   }
; CHECK:   ArgList (0x1002) {
; CHECK:     TypeLeafKind: LF_ARGLIST (0x1201)
; CHECK:     NumArgs: 0
; CHECK:     Arguments [
; CHECK:     ]
; CHECK:   }
; CHECK:   MemberFunction (0x1003) {
; CHECK:     TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK:     ReturnType: void (0x3)
; CHECK:     ClassType: A (0x1000)
; CHECK:     ThisType: A* (0x1001)
; CHECK:     CallingConvention: ThisCall (0xB)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 0
; CHECK:     ArgListType: () (0x1002)
; CHECK:     ThisAdjustment: 0
; CHECK:   }
; CHECK:   FieldList (0x1004) {
; CHECK:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:     OneMethod {
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: void A::() (0x1003)
; CHECK:       Name: A::thiscallcc
; CHECK:     }
; CHECK:   }
; CHECK:   Struct (0x1005) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 1
; CHECK:     Properties [ (0x0)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x1004)
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 1
; CHECK:     Name: A
; CHECK:   }
; CHECK:   StringId (0x1006) {
; CHECK:     TypeLeafKind: LF_STRING_ID (0x1605)
; CHECK:     Id: 0x0
; CHECK:     StringData: D:\src\llvm\build\t.cpp
; CHECK:   }
; CHECK:   UdtSourceLine (0x1007) {
; CHECK:     TypeLeafKind: LF_UDT_SRC_LINE (0x1606)
; CHECK:     UDT: A (0x1005)
; CHECK:     SourceFile: D:\src\llvm\build\t.cpp (0x1006)
; CHECK:     LineNumber: 1
; CHECK:   }
; CHECK:   MemberFuncId (0x1008) {
; CHECK:     TypeLeafKind: LF_MFUNC_ID (0x1602)
; CHECK:     ClassType: A (0x1000)
; CHECK:     FunctionType: void A::() (0x1003)
; CHECK:     Name: A::thiscallcc
; CHECK:   }
; CHECK:   Procedure (0x1009) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: void (0x3)
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 0
; CHECK:     ArgListType: () (0x1002)
; CHECK:   }
; CHECK:   FuncId (0x100A) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: void () (0x1009)
; CHECK:     Name: cdeclcc
; CHECK:   }
; CHECK:   Procedure (0x100B) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: void (0x3)
; CHECK:     CallingConvention: NearFast (0x4)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 0
; CHECK:     ArgListType: () (0x1002)
; CHECK:   }
; CHECK:   FuncId (0x100C) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: void () (0x100B)
; CHECK:     Name: fastcallcc
; CHECK:   }
; CHECK:   Procedure (0x100D) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: void (0x3)
; CHECK:     CallingConvention: NearStdCall (0x7)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 0
; CHECK:     ArgListType: () (0x1002)
; CHECK:   }
; CHECK:   FuncId (0x100E) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: void () (0x100D)
; CHECK:     Name: stdcallcc
; CHECK:   }
; CHECK:   Procedure (0x100F) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: void (0x3)
; CHECK:     CallingConvention: NearVector (0x18)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 0
; CHECK:     ArgListType: () (0x1002)
; CHECK:   }
; CHECK:   FuncId (0x1010) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: void () (0x100F)
; CHECK:     Name: vectorcallcc
; CHECK:   }
; CHECK: ]

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc19.0.23918"

%struct.A = type { i8 }

; Function Attrs: nounwind readnone
define x86_thiscallcc void @"\01?thiscallcc@A@@QAEXXZ"(%struct.A* nocapture %this) #0 align 2 !dbg !6 {
entry:
  tail call void @llvm.dbg.value(metadata %struct.A* %this, metadata !14, metadata !16), !dbg !17
  ret void, !dbg !18
}

; Function Attrs: norecurse nounwind readnone
define void @"\01?cdeclcc@@YAXXZ"() #1 !dbg !19 {
entry:
  ret void, !dbg !22
}

; Function Attrs: norecurse nounwind readnone
define x86_fastcallcc void @"\01?fastcallcc@@YIXXZ"() #1 !dbg !23 {
entry:
  ret void, !dbg !24
}

; Function Attrs: norecurse nounwind readnone
define x86_stdcallcc void @"\01?stdcallcc@@YGXXZ"() #1 !dbg !25 {
entry:
  ret void, !dbg !26
}

; Function Attrs: norecurse nounwind readnone
define x86_vectorcallcc void @"\01?vectorcallcc@@YQXXZ"() #1 !dbg !27 {
entry:
  ret void, !dbg !28
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind readnone "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { norecurse nounwind readnone "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 (trunk 272067)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 3.9.0 (trunk 272067)"}
!6 = distinct !DISubprogram(name: "A::thiscallcc", linkageName: "\01?thiscallcc@A@@QAEXXZ", scope: !7, file: !1, line: 4, type: !10, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !9, retainedNodes: !13)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !1, line: 1, size: 8, align: 8, elements: !8)
!8 = !{!9}
!9 = !DISubprogram(name: "A::thiscallcc", linkageName: "\01?thiscallcc@A@@QAEXXZ", scope: !7, file: !1, line: 2, type: !10, isLocal: false, isDefinition: false, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true)
!10 = !DISubroutineType(cc: DW_CC_BORLAND_thiscall, types: !11)
!11 = !{null, !12}
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 32, align: 32, flags: DIFlagArtificial | DIFlagObjectPointer)
!13 = !{!14}
!14 = !DILocalVariable(name: "this", arg: 1, scope: !6, type: !15, flags: DIFlagArtificial | DIFlagObjectPointer)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 32, align: 32)
!16 = !DIExpression()
!17 = !DILocation(line: 0, scope: !6)
!18 = !DILocation(line: 4, column: 23, scope: !6)
!19 = distinct !DISubprogram(name: "cdeclcc", linkageName: "\01?cdeclcc@@YAXXZ", scope: !1, file: !1, line: 5, type: !20, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!21 = !{null}
!22 = !DILocation(line: 5, column: 17, scope: !19)
!23 = distinct !DISubprogram(name: "fastcallcc", linkageName: "\01?fastcallcc@@YIXXZ", scope: !1, file: !1, line: 6, type: !29, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!24 = !DILocation(line: 6, column: 31, scope: !23)
!25 = distinct !DISubprogram(name: "stdcallcc", linkageName: "\01?stdcallcc@@YGXXZ", scope: !1, file: !1, line: 7, type: !30, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!26 = !DILocation(line: 7, column: 29, scope: !25)
!27 = distinct !DISubprogram(name: "vectorcallcc", linkageName: "\01?vectorcallcc@@YQXXZ", scope: !1, file: !1, line: 8, type: !31, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!28 = !DILocation(line: 8, column: 35, scope: !27)

!20 = !DISubroutineType(cc: DW_CC_normal, types: !21)
!29 = !DISubroutineType(cc: DW_CC_BORLAND_msfastcall, types: !21)
!30 = !DISubroutineType(cc: DW_CC_BORLAND_stdcall, types: !21)
!31 = !DISubroutineType(cc: DW_CC_LLVM_vectorcall, types: !21)
