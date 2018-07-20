; Verify "no source location" directives appear in appropriate places,
; but don't appear when we explicitly suppress them.
; RUN: llc < %s -o - | FileCheck %s
; RUN: llc < %s -o - -use-unknown-locations=Disable | FileCheck %s --check-prefix=DISABLE

; Generated from this .cpp targeting linux using -g
; and then removed function attributes as clutter.
;
; void bar(int *);
; void baz(int *);
; # 5 "no-source-loc.cpp"
; void foo(int x) {
;   int z;
;   if (x)
; # 20 "include.h"
;     bar(&z);
; # 10 "no-source-loc.cpp"
;   baz(&z);
; }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: uwtable
define void @_Z3fooi(i32 %x) !dbg !6 {
entry:
  %x.addr = alloca i32, align 4
  %z = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  %0 = load i32, i32* %x.addr, align 4, !dbg !8
  %tobool = icmp ne i32 %0, 0, !dbg !8
  br i1 %tobool, label %if.then, label %if.end, !dbg !8

if.then:                                          ; preds = %entry
  call void @_Z3barPi(i32* %z), !dbg !9
  br label %if.end, !dbg !9

if.end:                                           ; preds = %if.then, %entry
  call void @_Z3bazPi(i32* %z), !dbg !12
  ret void, !dbg !14
}

; CHECK:      .loc 1 7 7
; CHECK-NOT:  .loc
; CHECK:      .loc 1 0 7 is_stmt 0
; CHECK-NOT:  .loc
; CHECK:      .loc 2 20 5 is_stmt 1
; CHECK:      .LBB0_2:
; CHECK-NEXT: .loc 2 0 5 is_stmt 0
; CHECK-NOT:  .loc
; CHECK:      .loc 1 10 3 is_stmt 1
;
; DISABLE-NOT: .loc 1 0

declare void @_Z3barPi(i32*)

declare void @_Z3bazPi(i32*)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 4.0.0 (trunk 278782)", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2)
!1 = !DIFile(filename: "no-source-loc.cpp", directory: "/tests")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 4.0.0 (trunk 278782)"}
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 5, type: !7, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 7, column: 7, scope: !6)
!9 = !DILocation(line: 20, column: 5, scope: !10)
!10 = !DILexicalBlockFile(scope: !6, file: !11, discriminator: 0)
!11 = !DIFile(filename: "include.h", directory: "/tests")
!12 = !DILocation(line: 10, column: 3, scope: !13)
!13 = !DILexicalBlockFile(scope: !6, file: !1, discriminator: 0)
!14 = !DILocation(line: 11, column: 1, scope: !13)
