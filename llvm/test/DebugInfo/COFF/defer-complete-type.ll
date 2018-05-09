; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s

; Both A and B should get forward declarations and complete definitions for this
; example.

; C++ source to regenerate:
; $ cat t.cpp
; struct B { int b; };
; struct A { B *b; };
; int f(A *p) { return p->b->b; }
; $ clang t.cpp -S -emit-llvm -g -gcodeview -o t.ll

; CHECK: CodeViewTypes [
; CHECK:   Section: .debug$T (6)
; CHECK:   Magic: 0x4
; CHECK:   Struct (0x1000) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 0
; CHECK:     Properties [ (0x280)
; CHECK:       ForwardReference (0x80)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: 0x0
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 0
; CHECK:     Name: A
; CHECK:     LinkageName: .?AUA@@
; CHECK:   }
; CHECK:   Pointer (0x1001) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: A (0x1000)
; CHECK:     PointerAttributes: 0x1000C
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: Pointer (0x0)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     SizeOf: 8
; CHECK:   }
; CHECK:   ArgList (0x1002) {
; CHECK:     TypeLeafKind: LF_ARGLIST (0x1201)
; CHECK:     NumArgs: 1
; CHECK:     Arguments [
; CHECK:       ArgType: A* (0x1001)
; CHECK:     ]
; CHECK:   }
; CHECK:   Procedure (0x1003) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: int (0x74)
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 1
; CHECK:     ArgListType: (A*) (0x1002)
; CHECK:   }
; CHECK:   Struct (0x1004) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 0
; CHECK:     Properties [ (0x280)
; CHECK:       ForwardReference (0x80)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: 0x0
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 0
; CHECK:     Name: B
; CHECK:     LinkageName: .?AUB@@
; CHECK:   }
; CHECK:   Pointer (0x1005) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: B (0x1004)
; CHECK:     PointerAttributes: 0x1000C
; CHECK:     PtrType: Near64 (0xC)
; CHECK:     PtrMode: Pointer (0x0)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     SizeOf: 8
; CHECK:   }
; CHECK:   FieldList (0x1006) {
; CHECK:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:     DataMember {
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: B* (0x1005)
; CHECK:       FieldOffset: 0x0
; CHECK:       Name: b
; CHECK:     }
; CHECK:   }
; CHECK:   Struct (0x1007) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 1
; CHECK:     Properties [ (0x200)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x1006)
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 8
; CHECK:     Name: A
; CHECK:     LinkageName: .?AUA@@
; CHECK:   }
; CHECK:   StringId (0x1008) {
; CHECK:     TypeLeafKind: LF_STRING_ID (0x1605)
; CHECK:     Id: 0x0
; CHECK:     StringData: D:\src\llvm\build\t.cpp
; CHECK:   }
; CHECK:   UdtSourceLine (0x1009) {
; CHECK:     TypeLeafKind: LF_UDT_SRC_LINE (0x1606)
; CHECK:     UDT: A (0x1007)
; CHECK:     SourceFile: D:\src\llvm\build\t.cpp (0x1008)
; CHECK:     LineNumber: 2
; CHECK:   }
; CHECK:   FieldList (0x100A) {
; CHECK:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:     DataMember {
; CHECK:       AccessSpecifier: Public (0x3)
; CHECK:       Type: int (0x74)
; CHECK:       FieldOffset: 0x0
; CHECK:       Name: b
; CHECK:     }
; CHECK:   }
; CHECK:   Struct (0x100B) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 1
; CHECK:     Properties [ (0x200)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x100A)
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 4
; CHECK:     Name: B
; CHECK:     LinkageName: .?AUB@@
; CHECK:   }
; CHECK:   UdtSourceLine (0x100C) {
; CHECK:     TypeLeafKind: LF_UDT_SRC_LINE (0x1606)
; CHECK:     UDT: B (0x100B)
; CHECK:     SourceFile: D:\src\llvm\build\t.cpp (0x1008)
; CHECK:     LineNumber: 1
; CHECK:   }
; CHECK:   FuncId (0x100D) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: int (A*) (0x1003)
; CHECK:     Name: f
; CHECK:   }
; CHECK: ]

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.23918"

%struct.A = type { %struct.B* }
%struct.B = type { i32 }

; Function Attrs: nounwind uwtable
define i32 @"\01?f@@YAHPEAUA@@@Z"(%struct.A* %p) #0 !dbg !7 {
entry:
  %p.addr = alloca %struct.A*, align 8
  store %struct.A* %p, %struct.A** %p.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.A** %p.addr, metadata !19, metadata !20), !dbg !21
  %0 = load %struct.A*, %struct.A** %p.addr, align 8, !dbg !22
  %b = getelementptr inbounds %struct.A, %struct.A* %0, i32 0, i32 0, !dbg !23
  %1 = load %struct.B*, %struct.B** %b, align 8, !dbg !23
  %b1 = getelementptr inbounds %struct.B, %struct.B* %1, i32 0, i32 0, !dbg !24
  %2 = load i32, i32* %b1, align 4, !dbg !24
  ret i32 %2, !dbg !25
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

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
!7 = distinct !DISubprogram(name: "f", linkageName: "\01?f@@YAHPEAUA@@@Z", scope: !1, file: !1, line: 3, type: !8, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11}
!10 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64, align: 64)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !1, line: 2, size: 64, align: 64, elements: !13, identifier: ".?AUA@@")
!13 = !{!14}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !12, file: !1, line: 2, baseType: !15, size: 64, align: 64)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64, align: 64)
!16 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "B", file: !1, line: 1, size: 32, align: 32, elements: !17, identifier: ".?AUB@@")
!17 = !{!18}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !16, file: !1, line: 1, baseType: !10, size: 32, align: 32)
!19 = !DILocalVariable(name: "p", arg: 1, scope: !7, file: !1, line: 3, type: !11)
!20 = !DIExpression()
!21 = !DILocation(line: 3, column: 10, scope: !7)
!22 = !DILocation(line: 3, column: 22, scope: !7)
!23 = !DILocation(line: 3, column: 25, scope: !7)
!24 = !DILocation(line: 3, column: 28, scope: !7)
!25 = !DILocation(line: 3, column: 15, scope: !7)
