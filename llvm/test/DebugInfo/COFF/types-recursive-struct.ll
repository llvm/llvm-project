; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s

; This test ensures that circular type references through pointer types don't
; cause infinite recursion. It also tests that we always refer to the forward
; declaration type index in field lists and pointer types, which is consistent
; with what MSVC does. It ensures that these records get merged when merging
; streams even if the complete record types differ slightly due to ODR
; violations, i.e. methods that only exist ifndef NDEBUG.

; C++ source to regenerate:
; $ cat t.cpp
; struct B;
; struct A { B *b; };
; struct B { A a; };
; void f() {
;   A a;
;   B b;
; }
; $ clang t.cpp -S -emit-llvm -g -gcodeview -o t.ll

; CHECK: CodeViewTypes [
; CHECK:   Section: .debug$T (6)
; CHECK:   Magic: 0x4
; CHECK:   ArgList (0x1000) {
; CHECK:     TypeLeafKind: LF_ARGLIST (0x1201)
; CHECK:     NumArgs: 0
; CHECK:     Arguments [
; CHECK:     ]
; CHECK:   }
; CHECK:   Procedure (0x1001) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: void (0x3)
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 0
; CHECK:     ArgListType: () (0x1000)
; CHECK:   }
; CHECK:   FuncId (0x1002) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: void () (0x1001)
; CHECK:     Name: f
; CHECK:   }
; CHECK:   Struct (0x1003) {
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
; CHECK:   Struct (0x1004) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 0
; CHECK:     Properties [ (0x80)
; CHECK:       ForwardReference (0x80)
; CHECK:     ]
; CHECK:     FieldList: 0x0
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 0
; CHECK:     Name: B
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
; CHECK:     Properties [ (0x0)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x1006)
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 8
; CHECK:     Name: A
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
; CHECK:       Type: A (0x1003)
; CHECK:       FieldOffset: 0x0
; CHECK:       Name: a
; CHECK:     }
; CHECK:   }
; CHECK:   Struct (0x100B) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 1
; CHECK:     Properties [ (0x0)
; CHECK:     ]
; CHECK:     FieldList: <field list> (0x100A)
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 8
; CHECK:     Name: B
; CHECK:   }
; CHECK:   UdtSourceLine (0x100C) {
; CHECK:     TypeLeafKind: LF_UDT_SRC_LINE (0x1606)
; CHECK:     UDT: B (0x100B)
; CHECK:     SourceFile: D:\src\llvm\build\t.cpp (0x1008)
; CHECK:     LineNumber: 3
; CHECK:   }
; CHECK: ]

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.23918"

%struct.A = type { %struct.B* }
%struct.B = type { %struct.A }

; Function Attrs: nounwind uwtable
define void @"\01?f@@YAXXZ"() #0 !dbg !7 {
entry:
  %a = alloca %struct.A, align 8
  %b = alloca %struct.B, align 8
  call void @llvm.dbg.declare(metadata %struct.A* %a, metadata !10, metadata !18), !dbg !19
  call void @llvm.dbg.declare(metadata %struct.B* %b, metadata !20, metadata !18), !dbg !21
  ret void, !dbg !22
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
!7 = distinct !DISubprogram(name: "f", linkageName: "\01?f@@YAXXZ", scope: !1, file: !1, line: 4, type: !8, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocalVariable(name: "a", scope: !7, file: !1, line: 5, type: !11)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !1, line: 2, size: 64, align: 64, elements: !12)
!12 = !{!13}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !11, file: !1, line: 2, baseType: !14, size: 64, align: 64)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64, align: 64)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "B", file: !1, line: 3, size: 64, align: 64, elements: !16)
!16 = !{!17}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !15, file: !1, line: 3, baseType: !11, size: 64, align: 64)
!18 = !DIExpression()
!19 = !DILocation(line: 5, column: 5, scope: !7)
!20 = !DILocalVariable(name: "b", scope: !7, file: !1, line: 6, type: !15)
!21 = !DILocation(line: 6, column: 5, scope: !7)
!22 = !DILocation(line: 7, column: 1, scope: !7)
