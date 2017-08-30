; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s
source_filename = "test/DebugInfo/COFF/udts.ll"
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

; CHECK:      {{.*}}Proc{{.*}}Sym {
; CHECK:        DisplayName: f
; CHECK:        LinkageName: ?f@@YAXXZ
; CHECK:      }
; CHECK:      UDTSym {
; CHECK-NEXT:   Kind: S_UDT (0x1108)
; CHECK-NEXT:   Type: int (0x74)
; CHECK-NEXT:   UDTName: f::FOO
; CHECK-NEXT: }
; CHECK-NEXT: ProcEnd {

; CHECK:      {{.*}}Proc{{.*}}Sym {
; CHECK:        DisplayName: g
; CHECK:        LinkageName: ?g@@YAMPEAUS@@@Z
; CHECK:      }
; CHECK:      UDTSym {
; CHECK-NEXT:   Kind: S_UDT (0x1108)
; CHECK-NEXT:   Type: g::pun (0x{{[0-9A-F]+}})
; CHECK-NEXT:   UDTName: g::pun
; CHECK-NEXT: }
; CHECK-NEXT: ProcEnd {

; CHECK:      Subsection
; CHECK-NOT:  {{.*}}Proc{{.*}}Sym
; CHECK:      UDTSym {
; CHECK-NEXT:   Kind: S_UDT (0x1108)
; CHECK-NEXT: Type: S (0x{{[0-9A-F]+}})
; CHECK-NEXT: UDTName: S
; CHECK:      UDTSym {
; CHECK-NEXT:   Kind: S_UDT (0x1108)
; CHECK-NEXT: Type: <unnamed-tag> (0x{{[0-9A-F]+}})
; CHECK-NEXT: UDTName: U
; CHECK-NOT: UDTSym {

%struct.U = type { i32 }
%struct.S = type { i32 }
%union.pun = type { i32 }

@"\01?u@@3UU@@A" = global %struct.U zeroinitializer, align 4, !dbg !0

; Function Attrs: nounwind uwtable
define void @"\01?f@@YAXXZ"() #0 !dbg !15 {
entry:
  %f = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %f, metadata !18, metadata !20), !dbg !21
  ret void, !dbg !22
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind uwtable
define float @"\01?g@@YAMPEAUS@@@Z"(%struct.S* %s) #0 !dbg !23 {
entry:
  %s.addr = alloca %struct.S*, align 8
  %p = alloca %union.pun, align 4
  store %struct.S* %s, %struct.S** %s.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.S** %s.addr, metadata !31, metadata !20), !dbg !32
  call void @llvm.dbg.declare(metadata %union.pun* %p, metadata !33, metadata !20), !dbg !38
  %0 = load %struct.S*, %struct.S** %s.addr, align 8, !dbg !39
  %x = getelementptr inbounds %struct.S, %struct.S* %0, i32 0, i32 0, !dbg !40
  %1 = load i32, i32* %x, align 4, !dbg !40
  %x1 = bitcast %union.pun* %p to i32*, !dbg !41
  store i32 %1, i32* %x1, align 4, !dbg !42
  %f = bitcast %union.pun* %p to float*, !dbg !43
  %2 = load float, float* %f, align 4, !dbg !43
  ret float %2, !dbg !44
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12, !13}
!llvm.ident = !{!14}

!0 = distinct !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "u", linkageName: "\01?u@@3UU@@A", scope: !2, file: !3, line: 13, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 3.9.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_typedef, name: "U", file: !3, line: 12, baseType: !7)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !3, line: 12, size: 32, align: 32, elements: !8, identifier: ".?AUU@@")
!8 = !{!9}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !7, file: !3, line: 12, baseType: !10, size: 32, align: 32)
!10 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{i32 2, !"CodeView", i32 1}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"PIC Level", i32 2}
!14 = !{!"clang version 3.9.0 "}
!15 = distinct !DISubprogram(name: "f", linkageName: "\01?f@@YAXXZ", scope: !3, file: !3, line: 1, type: !16, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !2, variables: !4)
!16 = !DISubroutineType(types: !17)
!17 = !{null}
!18 = !DILocalVariable(name: "f", scope: !15, file: !3, line: 3, type: !19)
!19 = !DIDerivedType(tag: DW_TAG_typedef, name: "FOO", scope: !15, file: !3, line: 2, baseType: !10)
!20 = !DIExpression()
!21 = !DILocation(line: 3, column: 7, scope: !15)
!22 = !DILocation(line: 4, column: 1, scope: !15)
!23 = distinct !DISubprogram(name: "g", linkageName: "\01?g@@YAMPEAUS@@@Z", scope: !3, file: !3, line: 7, type: !24, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: false, unit: !2, variables: !4)
!24 = !DISubroutineType(types: !25)
!25 = !{!26, !27}
!26 = !DIBasicType(name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!27 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !28, size: 64, align: 64)
!28 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !3, line: 6, size: 32, align: 32, elements: !29, identifier: ".?AUS@@")
!29 = !{!30}
!30 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !28, file: !3, line: 6, baseType: !10, size: 32, align: 32)
!31 = !DILocalVariable(name: "s", arg: 1, scope: !23, file: !3, line: 7, type: !27)
!32 = !DILocation(line: 7, column: 12, scope: !23)
!33 = !DILocalVariable(name: "p", scope: !23, file: !3, line: 8, type: !34)
!34 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "pun", scope: !23, file: !3, line: 8, size: 32, align: 32, elements: !35)
!35 = !{!36, !37}
!36 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !34, file: !3, line: 8, baseType: !10, size: 32, align: 32)
!37 = !DIDerivedType(tag: DW_TAG_member, name: "f", scope: !34, file: !3, line: 8, baseType: !26, size: 32, align: 32)
!38 = !DILocation(line: 8, column: 33, scope: !23)
!39 = !DILocation(line: 9, column: 9, scope: !23)
!40 = !DILocation(line: 9, column: 12, scope: !23)
!41 = !DILocation(line: 9, column: 5, scope: !23)
!42 = !DILocation(line: 9, column: 7, scope: !23)
!43 = !DILocation(line: 10, column: 12, scope: !23)
!44 = !DILocation(line: 10, column: 3, scope: !23)

