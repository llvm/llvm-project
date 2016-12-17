; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.0"

; C++ source to regenerate:
; $ cat t.cpp
; void f() {
;   typedef int FOO;
;   FOO f;
; }
;
; struct S { int x; };
; float g(S *s) {
;   union pun { int x; float f; } p;
;   p.x = s->x;
;   return p.f;
; }
; typedef struct { int x; } U;
; U u;

; CHECK:      ProcStart {
; CHECK:        DisplayName: f
; CHECK:        LinkageName: ?f@@YAXXZ
; CHECK:      }
; CHECK:      UDT {
; CHECK-NEXT:   Type: int (0x74)
; CHECK-NEXT:   UDTName: f::FOO
; CHECK-NEXT: }
; CHECK-NEXT: ProcEnd {
; CHECK-NEXT: }

; CHECK:      ProcStart {
; CHECK:        DisplayName: g
; CHECK:        LinkageName: ?g@@YAMPEAUS@@@Z
; CHECK:      }
; CHECK:      UDT {
; CHECK-NEXT:   Type: g::pun (0x{{[0-9A-F]+}})
; CHECK-NEXT:   UDTName: g::pun
; CHECK-NEXT: }
; CHECK-NEXT: ProcEnd {
; CHECK-NEXT: }

; CHECK:      Subsection
; CHECK-NOT:  ProcStart
; CHECK:      UDT {
; CHECK-NEXT: Type: S (0x{{[0-9A-F]+}})
; CHECK-NEXT: UDTName: S
; CHECK:      UDT {
; CHECK-NEXT: Type: <unnamed-tag> (0x{{[0-9A-F]+}})
; CHECK-NEXT: UDTName: U
; CHECK-NOT: UDT {


%struct.U = type { i32 }
%struct.S = type { i32 }
%union.pun = type { i32 }

@"\01?u@@3UU@@A" = global %struct.U zeroinitializer, align 4, !dbg !4

; Function Attrs: nounwind uwtable
define void @"\01?f@@YAXXZ"() #0 !dbg !14 {
entry:
  %f = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %f, metadata !17, metadata !19), !dbg !20
  ret void, !dbg !21
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind uwtable
define float @"\01?g@@YAMPEAUS@@@Z"(%struct.S* %s) #0 !dbg !22 {
entry:
  %s.addr = alloca %struct.S*, align 8
  %p = alloca %union.pun, align 4
  store %struct.S* %s, %struct.S** %s.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.S** %s.addr, metadata !30, metadata !19), !dbg !31
  call void @llvm.dbg.declare(metadata %union.pun* %p, metadata !32, metadata !19), !dbg !37
  %0 = load %struct.S*, %struct.S** %s.addr, align 8, !dbg !38
  %x = getelementptr inbounds %struct.S, %struct.S* %0, i32 0, i32 0, !dbg !39
  %1 = load i32, i32* %x, align 4, !dbg !39
  %x1 = bitcast %union.pun* %p to i32*, !dbg !40
  store i32 %1, i32* %x1, align 4, !dbg !41
  %f = bitcast %union.pun* %p to float*, !dbg !42
  %2 = load float, float* %f, align 4, !dbg !42
  ret float %2, !dbg !43
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11, !12}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!2 = !{}
!3 = !{!4}
!4 = distinct !DIGlobalVariable(name: "u", linkageName: "\01?u@@3UU@@A", scope: !0, file: !1, line: 13, type: !5, isLocal: false, isDefinition: true)
!5 = !DIDerivedType(tag: DW_TAG_typedef, name: "U", file: !1, line: 12, baseType: !6)
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !1, line: 12, size: 32, align: 32, elements: !7, identifier: ".?AUU@@")
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !6, file: !1, line: 12, baseType: !9, size: 32, align: 32)
!9 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !{i32 2, !"CodeView", i32 1}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"PIC Level", i32 2}
!13 = !{!"clang version 3.9.0 "}
!14 = distinct !DISubprogram(name: "f", linkageName: "\01?f@@YAXXZ", scope: !1, file: !1, line: 1, type: !15, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!15 = !DISubroutineType(types: !16)
!16 = !{null}
!17 = !DILocalVariable(name: "f", scope: !14, file: !1, line: 3, type: !18)
!18 = !DIDerivedType(tag: DW_TAG_typedef, name: "FOO", scope: !14, file: !1, line: 2, baseType: !9)
!19 = !DIExpression()
!20 = !DILocation(line: 3, column: 7, scope: !14)
!21 = !DILocation(line: 4, column: 1, scope: !14)
!22 = distinct !DISubprogram(name: "g", linkageName: "\01?g@@YAMPEAUS@@@Z", scope: !1, file: !1, line: 7, type: !23, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!23 = !DISubroutineType(types: !24)
!24 = !{!25, !26}
!25 = !DIBasicType(name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !27, size: 64, align: 64)
!27 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !1, line: 6, size: 32, align: 32, elements: !28, identifier: ".?AUS@@")
!28 = !{!29}
!29 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !27, file: !1, line: 6, baseType: !9, size: 32, align: 32)
!30 = !DILocalVariable(name: "s", arg: 1, scope: !22, file: !1, line: 7, type: !26)
!31 = !DILocation(line: 7, column: 12, scope: !22)
!32 = !DILocalVariable(name: "p", scope: !22, file: !1, line: 8, type: !33)
!33 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "pun", scope: !22, file: !1, line: 8, size: 32, align: 32, elements: !34)
!34 = !{!35, !36}
!35 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !33, file: !1, line: 8, baseType: !9, size: 32, align: 32)
!36 = !DIDerivedType(tag: DW_TAG_member, name: "f", scope: !33, file: !1, line: 8, baseType: !25, size: 32, align: 32)
!37 = !DILocation(line: 8, column: 33, scope: !22)
!38 = !DILocation(line: 9, column: 9, scope: !22)
!39 = !DILocation(line: 9, column: 12, scope: !22)
!40 = !DILocation(line: 9, column: 5, scope: !22)
!41 = !DILocation(line: 9, column: 7, scope: !22)
!42 = !DILocation(line: 10, column: 12, scope: !22)
!43 = !DILocation(line: 10, column: 3, scope: !22)
