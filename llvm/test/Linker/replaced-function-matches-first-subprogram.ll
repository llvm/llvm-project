; RUN: llvm-link %s %S/Inputs/replaced-function-matches-first-subprogram.ll -S | FileCheck %s

; Generated from C++ source:
;
; // repro/t.h
; template <class T> struct Class {
;   int foo() { return 0; }
; };
; // repro/d1/t1.cpp
; #include "t.h"
; int foo() { return Class<int>().foo(); }
; // repro/d2/t2.cpp
; #include "t.h"
; template struct Class<int>;

%struct.Class = type { i8 }

; CHECK: define i32 @_Z3foov(){{.*}} !dbg ![[SP1:[0-9]+]]
define i32 @_Z3foov() !dbg !4 {
entry:
  %tmp = alloca %struct.Class, align 1
  %call = call i32 @_ZN5ClassIiE3fooEv(%struct.Class* %tmp), !dbg !14
  ret i32 %call, !dbg !14
}

; CHECK: define weak_odr i32 @_ZN5ClassIiE3fooEv(%struct.Class* %this){{.*}} !dbg ![[SP2:[0-9]+]] {
; CHECK-NOT: }
; CHECK: !dbg ![[LOC:[0-9]+]]
define linkonce_odr i32 @_ZN5ClassIiE3fooEv(%struct.Class* %this) align 2 !dbg !7 {
entry:
  %this.addr = alloca %struct.Class*, align 8
  store %struct.Class* %this, %struct.Class** %this.addr, align 8
  %this1 = load %struct.Class*, %struct.Class** %this.addr
  ret i32 0, !dbg !15
}

; CHECK: !llvm.dbg.cu = !{![[CU1:[0-9]+]], ![[CU2:[0-9]+]]}
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11, !12}
!llvm.ident = !{!13}

; Extract out the list of subprograms from each compile unit.
; CHECK: ![[CU1:[0-9]+]] = distinct !DICompileUnit(
; CHECK: ![[CU2:[0-9]+]] = distinct !DICompileUnit(
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.6.0 (trunk 224193) (llvm/trunk 224197)", isOptimized: false, emissionKind: LineTablesOnly, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "t1.cpp", directory: "/Users/dexonsmith/data/llvm/staging/test/Linker/repro/d1")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", line: 2, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 2, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "t1.cpp", directory: "/Users/dexonsmith/data/llvm/staging/test/Linker/repro/d1")
!6 = !DISubroutineType(types: !2)

; Extract out the file from the replaced subprogram.
; CHECK-DAG: ![[SP2:.*]] = distinct !DISubprogram({{.*}} file: ![[FILE:[0-9]+]],{{.*}}, unit: ![[CU2]]

; We can't use CHECK-NOT/CHECK-SAME with a CHECK-DAG, so rely on field order to
; prove that there's no function: here.
; CHECK-DAG: ![[SP2r:.*]] = {{.*}}!DISubprogram({{.*}} isOptimized: false, unit: ![[CU1]], retainedNodes:
!7 = distinct !DISubprogram(name: "foo", line: 2, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 2, file: !8, scope: !9, type: !6, retainedNodes: !2)

; The new subprogram should be pointing at the new directory.
; CHECK-DAG: ![[FILE]] = !DIFile(filename: "../t.h", directory: "/Users/dexonsmith/data/llvm/staging/test/Linker/repro/d2")
!8 = !DIFile(filename: "../t.h", directory: "/Users/dexonsmith/data/llvm/staging/test/Linker/repro/d1")
!9 = !DIFile(filename: "../t.h", directory: "/Users/dexonsmith/data/llvm/staging/test/Linker/repro/d1")
!10 = !{i32 2, !"Dwarf Version", i32 2}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"PIC Level", i32 2}
!13 = !{!"clang version 3.6.0 (trunk 224193) (llvm/trunk 224197)"}
!14 = !DILocation(line: 2, column: 20, scope: !4)

; The same subprogram should be pointed to by inside the !dbg reference.
; CHECK: ![[LOC]] = !DILocation(line: 2, column: 15, scope: ![[SP2]])
!15 = !DILocation(line: 2, column: 15, scope: !7)
