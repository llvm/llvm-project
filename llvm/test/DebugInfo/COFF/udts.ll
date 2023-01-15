; RUN: llc < %s -filetype=obj > %t.obj
; RUN: llvm-readobj --codeview %t.obj | FileCheck --check-prefix=READOBJ %s
; RUN: llvm-pdbutil dump -symbols %t.obj | FileCheck --check-prefix=PDBUTIL %s

; C++ to regenerate:
; $ clang -g -gcodeview -m64 -S -emit-llvm t.cpp
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
; struct A {
;   // We should not output S_UDT for nested typedef.
;   typedef S C;
;   C c;
;   // We should output S_UDT for typedef of nested unnamed struct
;   typedef struct { long X; } D;
;   D d;
; };
; A a;
; 
; typedef struct { int x; } U;
; U u;

; READOBJ-NOT:  UDTName: A::C
; READOBJ:      {{.*}}Proc{{.*}}Sym {
; READOBJ:        DisplayName: f
; READOBJ:        LinkageName: ?f@@YAXXZ
; READOBJ:      }
; READOBJ:      UDTSym {
; READOBJ-NEXT:   Kind: S_UDT (0x1108)
; READOBJ-NEXT:   Type: int (0x74)
; READOBJ-NEXT:   UDTName: f::FOO
; READOBJ-NEXT: }
; READOBJ-NEXT: ProcEnd {

; READOBJ:      {{.*}}Proc{{.*}}Sym {
; READOBJ:        DisplayName: g
; READOBJ:        LinkageName: ?g@@YAMPEAUS@@@Z
; READOBJ:      }
; READOBJ:      UDTSym {
; READOBJ-NEXT:   Kind: S_UDT (0x1108)
; READOBJ-NEXT:   Type: g::pun (0x{{[0-9A-F]+}})
; READOBJ-NEXT:   UDTName: g::pun
; READOBJ-NEXT: }
; READOBJ-NEXT: ProcEnd {

; READOBJ:      Subsection
; READOBJ-NOT:  {{.*}}Proc{{.*}}Sym
; READOBJ:      UDTSym {
; READOBJ-NEXT:   Kind: S_UDT (0x1108)
; READOBJ-NEXT: Type: S (0x{{[0-9A-F]+}})
; READOBJ-NEXT: UDTName: S
; READOBJ:      UDTSym {
; READOBJ-NEXT:   Kind: S_UDT (0x1108)
; READOBJ-NEXT: Type: A (0x{{[0-9A-F]+}})
; READOBJ-NEXT: UDTName: A
; READOBJ:      UDTSym {
; READOBJ-NEXT:   Kind: S_UDT (0x1108)
; READOBJ-NEXT: Type: A::D (0x{{[0-9A-F]+}})
; READOBJ-NEXT: UDTName: A::D
; READOBJ:      UDTSym {
; READOBJ-NEXT:   Kind: S_UDT (0x1108)
; READOBJ-NEXT: Type: U (0x{{[0-9A-F]+}})
; READOBJ-NEXT: UDTName: U
; READOBJ:      UDTSym {
; READOBJ-NEXT:   Kind: S_UDT (0x1108)
; READOBJ-NEXT: Type: U (0x{{[0-9A-F]+}})
; READOBJ-NEXT: UDTName: U
; READOBJ-NOT: UDTSym

; PDBUTIL:                           Symbols
; PDBUTIL-NEXT: ============================================================
; PDBUTIL-NOT:   S_UDT {{.*}} `A::C`
; PDBUTIL:       S_UDT [size = 16] `f::FOO`
; PDBUTIL:       S_UDT [size = 16] `g::pun`
; PDBUTIL:       S_UDT [size = 12] `S`
; PDBUTIL:       S_UDT [size = 12] `A`
; PDBUTIL:       S_UDT [size = 16] `A::D`
; PDBUTIL:       S_UDT [size = 12] `U`
; PDBUTIL:       S_UDT [size = 12] `U`

source_filename = "test/DebugInfo/COFF/udts.ll"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.11.25506"

%struct.A = type { %struct.S, %"struct.A::D" }
%struct.S = type { i32 }
%"struct.A::D" = type { i32 }
%struct.U = type { i32 }
%union.pun = type { i32 }

@"\01?a@@3UA@@A" = global %struct.A zeroinitializer, align 4, !dbg !0
@"\01?u@@3UU@@A" = global %struct.U zeroinitializer, align 4, !dbg !6

; Function Attrs: noinline nounwind optnone uwtable
define void @"\01?f@@YAXXZ"() #0 !dbg !31 {
  %1 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata ptr %1, metadata !34, metadata !DIExpression()), !dbg !36
  ret void, !dbg !37
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone uwtable
define float @"\01?g@@YAMPEAUS@@@Z"(ptr) #0 !dbg !38 {
  %2 = alloca ptr, align 8
  %3 = alloca %union.pun, align 4
  store ptr %0, ptr %2, align 8
  call void @llvm.dbg.declare(metadata ptr %2, metadata !43, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.declare(metadata ptr %3, metadata !45, metadata !DIExpression()), !dbg !50
  %4 = load ptr, ptr %2, align 8, !dbg !51
  %5 = load i32, ptr %4, align 4, !dbg !52
  store i32 %5, ptr %3, align 4, !dbg !54
  %6 = load float, ptr %3, align 4, !dbg !55
  ret float %6, !dbg !56
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!26, !27, !28, !29}
!llvm.ident = !{!30}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", linkageName: "\01?a@@3UA@@A", scope: !2, file: !3, line: 21, type: !13, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 6.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvmbuild\5Cninja-release", checksumkind: CSK_MD5, checksum: "e894de94ed2e0d503ebb5dbcc550c544")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "u", linkageName: "\01?u@@3UU@@A", scope: !2, file: !3, line: 24, type: !8, isLocal: false, isDefinition: true)
!8 = !DIDerivedType(tag: DW_TAG_typedef, name: "U", file: !3, line: 23, baseType: !9)
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "U", file: !3, line: 23, size: 32, elements: !10, identifier: ".?AUU@@")
!10 = !{!11}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !9, file: !3, line: 23, baseType: !12, size: 32)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !3, line: 13, size: 64, elements: !14, identifier: ".?AUA@@")
!14 = !{!15, !19, !20, !24, !25}
!15 = !DIDerivedType(tag: DW_TAG_typedef, name: "C", scope: !13, file: !3, line: 15, baseType: !16)
!16 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !3, line: 6, size: 32, elements: !17, identifier: ".?AUS@@")
!17 = !{!18}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !16, file: !3, line: 6, baseType: !12, size: 32)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !13, file: !3, line: 16, baseType: !15, size: 32)
!20 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "D", scope: !13, file: !3, line: 18, size: 32, elements: !21, identifier: ".?AUD@A@@")
!21 = !{!22}
!22 = !DIDerivedType(tag: DW_TAG_member, name: "X", scope: !20, file: !3, line: 18, baseType: !23, size: 32)
!23 = !DIBasicType(name: "long int", size: 32, encoding: DW_ATE_signed)
!24 = !DIDerivedType(tag: DW_TAG_typedef, name: "D", scope: !13, file: !3, line: 18, baseType: !20)
!25 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !13, file: !3, line: 19, baseType: !24, size: 32, offset: 32)
!26 = !{i32 2, !"CodeView", i32 1}
!27 = !{i32 2, !"Debug Info Version", i32 3}
!28 = !{i32 1, !"wchar_size", i32 2}
!29 = !{i32 7, !"PIC Level", i32 2}
!30 = !{!"clang version 6.0.0 "}
!31 = distinct !DISubprogram(name: "f", linkageName: "\01?f@@YAXXZ", scope: !3, file: !3, line: 1, type: !32, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!32 = !DISubroutineType(types: !33)
!33 = !{null}
!34 = !DILocalVariable(name: "f", scope: !31, file: !3, line: 3, type: !35)
!35 = !DIDerivedType(tag: DW_TAG_typedef, name: "FOO", scope: !31, file: !3, line: 2, baseType: !12)
!36 = !DILocation(line: 3, column: 7, scope: !31)
!37 = !DILocation(line: 4, column: 1, scope: !31)
!38 = distinct !DISubprogram(name: "g", linkageName: "\01?g@@YAMPEAUS@@@Z", scope: !3, file: !3, line: 7, type: !39, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!39 = !DISubroutineType(types: !40)
!40 = !{!41, !42}
!41 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!42 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!43 = !DILocalVariable(name: "s", arg: 1, scope: !38, file: !3, line: 7, type: !42)
!44 = !DILocation(line: 7, column: 12, scope: !38)
!45 = !DILocalVariable(name: "p", scope: !38, file: !3, line: 8, type: !46)
!46 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "pun", scope: !38, file: !3, line: 8, size: 32, elements: !47)
!47 = !{!48, !49}
!48 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !46, file: !3, line: 8, baseType: !12, size: 32)
!49 = !DIDerivedType(tag: DW_TAG_member, name: "f", scope: !46, file: !3, line: 8, baseType: !41, size: 32)
!50 = !DILocation(line: 8, column: 33, scope: !38)
!51 = !DILocation(line: 9, column: 9, scope: !38)
!52 = !DILocation(line: 9, column: 12, scope: !38)
!53 = !DILocation(line: 9, column: 5, scope: !38)
!54 = !DILocation(line: 9, column: 7, scope: !38)
!55 = !DILocation(line: 10, column: 12, scope: !38)
!56 = !DILocation(line: 10, column: 3, scope: !38)
