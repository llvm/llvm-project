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


%struct.S = type { i32 }
%union.pun = type { i32 }

; Function Attrs: nounwind uwtable
define void @"\01?f@@YAXXZ"() #0 !dbg !7 {
entry:
  %f = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %f, metadata !10, metadata !13), !dbg !14
  ret void, !dbg !15
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind uwtable
define float @"\01?g@@YAMPEAUS@@@Z"(%struct.S* %s) #0 !dbg !16 {
entry:
  %s.addr = alloca %struct.S*, align 8
  %p = alloca %union.pun, align 4
  store %struct.S* %s, %struct.S** %s.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.S** %s.addr, metadata !24, metadata !13), !dbg !25
  call void @llvm.dbg.declare(metadata %union.pun* %p, metadata !26, metadata !13), !dbg !31
  %0 = load %struct.S*, %struct.S** %s.addr, align 8, !dbg !32
  %x = getelementptr inbounds %struct.S, %struct.S* %0, i32 0, i32 0, !dbg !33
  %1 = load i32, i32* %x, align 4, !dbg !33
  %x1 = bitcast %union.pun* %p to i32*, !dbg !34
  store i32 %1, i32* %x1, align 4, !dbg !35
  %f = bitcast %union.pun* %p to float*, !dbg !36
  %2 = load float, float* %f, align 4, !dbg !36
  ret float %2, !dbg !37
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 (trunk 273566) (llvm/trunk 273570)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.cpp", directory: "/usr/local/google/work/llvm/build.release")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 3.9.0 (trunk 273566) (llvm/trunk 273570)"}
!7 = distinct !DISubprogram(name: "f", linkageName: "\01?f@@YAXXZ", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocalVariable(name: "f", scope: !7, file: !1, line: 3, type: !11)
!11 = !DIDerivedType(tag: DW_TAG_typedef, name: "FOO", scope: !7, file: !1, line: 2, baseType: !12)
!12 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = !DIExpression()
!14 = !DILocation(line: 3, column: 7, scope: !7)
!15 = !DILocation(line: 4, column: 1, scope: !7)
!16 = distinct !DISubprogram(name: "g", linkageName: "\01?g@@YAMPEAUS@@@Z", scope: !1, file: !1, line: 7, type: !17, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!17 = !DISubroutineType(types: !18)
!18 = !{!19, !20}
!19 = !DIBasicType(name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !21, size: 64, align: 64)
!21 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !1, line: 6, size: 32, align: 32, elements: !22, identifier: ".?AUS@@")
!22 = !{!23}
!23 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !21, file: !1, line: 6, baseType: !12, size: 32, align: 32)
!24 = !DILocalVariable(name: "s", arg: 1, scope: !16, file: !1, line: 7, type: !20)
!25 = !DILocation(line: 7, column: 12, scope: !16)
!26 = !DILocalVariable(name: "p", scope: !16, file: !1, line: 8, type: !27)
!27 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "pun", scope: !16, file: !1, line: 8, size: 32, align: 32, elements: !28)
!28 = !{!29, !30}
!29 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !27, file: !1, line: 8, baseType: !12, size: 32, align: 32)
!30 = !DIDerivedType(tag: DW_TAG_member, name: "f", scope: !27, file: !1, line: 8, baseType: !19, size: 32, align: 32)
!31 = !DILocation(line: 8, column: 33, scope: !16)
!32 = !DILocation(line: 9, column: 9, scope: !16)
!33 = !DILocation(line: 9, column: 12, scope: !16)
!34 = !DILocation(line: 9, column: 5, scope: !16)
!35 = !DILocation(line: 9, column: 7, scope: !16)
!36 = !DILocation(line: 10, column: 12, scope: !16)
!37 = !DILocation(line: 10, column: 3, scope: !16)
