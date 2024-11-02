; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s
; RUN: llc < %s | llvm-mc -filetype=obj --triple=x86_64-windows | llvm-readobj - --codeview | FileCheck %s

; This test checks that types which are used in expressions, but for which
; there are no variables, known as retained types, get emitted.

; C++ source to regenerate:
; $ cat /tmp/a.cc
; struct S { int x; };
; int f(ptr p) {
;   return static_cast<S*>(p)->x;
; }
; $ clang /tmp/a.cc -S -emit-llvm -g -gcodeview -target x86_64-windows-msvc -o t.ll

; CHECK:       Struct (0x{{[0-9A-F]+}}) {
; CHECK-NEXT:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK-NEXT:     MemberCount: 0
; CHECK-NEXT:     Properties [ (0x280)
; CHECK-NEXT:       ForwardReference (0x80)
; CHECK-NEXT:       HasUniqueName (0x200)
; CHECK-NEXT:     ]
; CHECK-NEXT:     FieldList: 0x0
; CHECK-NEXT:     DerivedFrom: 0x0
; CHECK-NEXT:     VShape: 0x0
; CHECK-NEXT:     SizeOf: 0
; CHECK-NEXT:     Name: S
; CHECK-NEXT:     LinkageName: .?AUS@@
; CHECK-NEXT:   }

; CHECK:        Struct (0x{{[0-9A-F]+}}) {
; CHECK-NEXT:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK-NEXT:     MemberCount: 1
; CHECK-NEXT:     Properties [ (0x200)
; CHECK-NEXT:       HasUniqueName (0x200)
; CHECK-NEXT:     ]
; CHECK-NEXT:     FieldList: <field list> (0x{{[0-9A-F]+}})
; CHECK-NEXT:     DerivedFrom: 0x0
; CHECK-NEXT:     VShape: 0x0
; CHECK-NEXT:     SizeOf: 4
; CHECK-NEXT:     Name: S
; CHECK-NEXT:     LinkageName: .?AUS@@
; CHECK-NEXT:   }

; ModuleID = '/tmp/a.cc'
source_filename = "/tmp/a.cc"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64--windows-msvc18.0.0"

%struct.S = type { i32 }

; Function Attrs: nounwind uwtable
define i32 @"\01?f@@YAHPEAX@Z"(ptr %p) #0 !dbg !13 {
entry:
  %p.addr = alloca ptr, align 8
  store ptr %p, ptr %p.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %p.addr, metadata !17, metadata !18), !dbg !19
  %0 = load ptr, ptr %p.addr, align 8, !dbg !20
  %1 = load i32, ptr %0, align 4, !dbg !22
  ret i32 %1, !dbg !23
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 (trunk 273441) (llvm/trunk 273449)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3)
!1 = !DIFile(filename: "/tmp/a.cc", directory: "/usr/local/google/work/llvm/build.release")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64, align: 64)
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !1, line: 1, size: 32, align: 32, elements: !6, identifier: ".?AUS@@")
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !5, file: !1, line: 1, baseType: !8, size: 32, align: 32)
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"CodeView", i32 1}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"PIC Level", i32 2}
!12 = !{!"clang version 3.9.0 (trunk 273441) (llvm/trunk 273449)"}
!13 = distinct !DISubprogram(name: "f", linkageName: "\01?f@@YAHPEAX@Z", scope: !1, file: !1, line: 2, type: !14, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!14 = !DISubroutineType(types: !15)
!15 = !{!8, !16}
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64, align: 64)
!17 = !DILocalVariable(name: "p", arg: 1, scope: !13, file: !1, line: 2, type: !16)
!18 = !DIExpression()
!19 = !DILocation(line: 2, column: 13, scope: !13)
!20 = !DILocation(line: 3, column: 26, scope: !13)
!21 = !DILocation(line: 3, column: 10, scope: !13)
!22 = !DILocation(line: 3, column: 30, scope: !13)
!23 = !DILocation(line: 3, column: 3, scope: !13)
