; RUN: opt < %s -passes=always-inline -S | FileCheck %s
; RUN: opt -passes='always-inline' -S < %s | FileCheck %s

; Original input generated from clang -emit-llvm -S -c -mllvm -disable-llvm-optzns
;
; #define CALLS1 f2(); f2();
; #define CALLS2 f4(); f4();
; void f1();
; inline __attribute__((always_inline)) void f2() {
;   f1();
; }
; inline __attribute__((always_inline)) void f3() {
;   CALLS1
; }
; inline __attribute__((always_inline)) void f4() {
;   f3();
; }
; void f() {
;   CALLS2
; }

; There should be unique locations for all 4 of these instructions, correctly
; describing the inlining that has occurred, even in the face of duplicate call
; site locations.

; The nomenclature used for the tags here is <function name>[cs<number>] where
; 'cs' is an abbreviation for 'call site' and the number indicates which call
; site from within the named function this is. (so, given the above inlining, we
; should have 4 calls to 'f1', two from the first call to f4 and two from the
; second call to f4)

; CHECK: call void @_Z2f1v(), !dbg [[fcs1_f4_f3cs1_f2:![0-9]+]]
; CHECK: call void @_Z2f1v(), !dbg [[fcs1_f4_f3cs2_f2:![0-9]+]]
; CHECK: call void @_Z2f1v(), !dbg [[fcs2_f4_f3cs1_f2:![0-9]+]]
; CHECK: call void @_Z2f1v(), !dbg [[fcs2_f4_f3cs2_f2:![0-9]+]]

; CHECK-DAG: [[F:![0-9]+]]  = distinct !DISubprogram(name: "f"
; CHECK-DAG: [[F2:![0-9]+]] = distinct !DISubprogram(name: "f2"
; CHECK-DAG: [[F3:![0-9]+]] = distinct !DISubprogram(name: "f3"
; CHECK-DAG: [[F4:![0-9]+]] = distinct !DISubprogram(name: "f4"

; CHECK-DAG: [[fcs1_f4_f3cs1_f2]] = {{.*}}, scope: [[F2]], inlinedAt: [[fcs1_f4_f3cs1:![0-9]+]])
; CHECK-DAG: [[fcs1_f4_f3cs1]] = {{.*}}, scope: [[F3]], inlinedAt: [[fcs1_f4:![0-9]+]])
; CHECK-DAG: [[fcs1_f4]] = {{.*}}, scope: [[F4]], inlinedAt: [[fcs1:![0-9]+]])
; CHECK-DAG: [[fcs1]] = {{.*}}, scope: [[F]])
; CHECK-DAG: [[fcs1_f4_f3cs2_f2]] = {{.*}}, scope: [[F2]], inlinedAt: [[fcs1_f4_f3cs2:![0-9]+]])
; CHECK-DAG: [[fcs1_f4_f3cs2]] = {{.*}}, scope: [[F3]], inlinedAt: [[fcs1_f4]])

; CHECK-DAG: [[fcs2_f4_f3cs1_f2]] = {{.*}}, scope: [[F2]], inlinedAt: [[fcs2_f4_f3cs1:![0-9]+]])
; CHECK-DAG: [[fcs2_f4_f3cs1]] = {{.*}}, scope: [[F3]], inlinedAt: [[fcs2_f4:![0-9]+]])
; CHECK-DAG: [[fcs2_f4]] = {{.*}}, scope: [[F4]], inlinedAt: [[fcs2:![0-9]+]])
; CHECK-DAG: [[fcs2]] = {{.*}}, scope: [[F]])
; CHECK-DAG: [[fcs2_f4_f3cs2_f2]] = {{.*}}, scope: [[F2]], inlinedAt: [[fcs2_f4_f3cs2:![0-9]+]])
; CHECK-DAG: [[fcs2_f4_f3cs2]] = {{.*}}, scope: [[F3]], inlinedAt: [[fcs2_f4]])

$_Z2f4v = comdat any

$_Z2f3v = comdat any

$_Z2f2v = comdat any

; Function Attrs: uwtable
define void @_Z1fv() #0 !dbg !4 {
entry:
  call void @_Z2f4v(), !dbg !13
  call void @_Z2f4v(), !dbg !13
  ret void, !dbg !14
}

; Function Attrs: alwaysinline inlinehint uwtable
define linkonce_odr void @_Z2f4v() #1 comdat !dbg !7 {
entry:
  call void @_Z2f3v(), !dbg !15
  ret void, !dbg !16
}

; Function Attrs: alwaysinline inlinehint uwtable
define linkonce_odr void @_Z2f3v() #1 comdat !dbg !8 {
entry:
  call void @_Z2f2v(), !dbg !17
  call void @_Z2f2v(), !dbg !17
  ret void, !dbg !18
}

; Function Attrs: alwaysinline inlinehint uwtable
define linkonce_odr void @_Z2f2v() #1 comdat !dbg !9 {
entry:
  call void @_Z2f1v(), !dbg !19
  ret void, !dbg !20
}

declare void @_Z2f1v() #2

attributes #0 = { uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { alwaysinline inlinehint uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.7.0 (trunk 226474) (llvm/trunk 226478)", isOptimized: false, emissionKind: LineTablesOnly, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "debug-info-duplicate-calls.cpp", directory: "/tmp/dbginfo")
!2 = !{}
!4 = distinct !DISubprogram(name: "f", line: 13, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 13, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "debug-info-duplicate-calls.cpp", directory: "/tmp/dbginfo")
!6 = !DISubroutineType(types: !2)
!7 = distinct !DISubprogram(name: "f4", line: 10, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 10, file: !1, scope: !5, type: !6, retainedNodes: !2)
!8 = distinct !DISubprogram(name: "f3", line: 7, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 7, file: !1, scope: !5, type: !6, retainedNodes: !2)
!9 = distinct !DISubprogram(name: "f2", line: 4, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 4, file: !1, scope: !5, type: !6, retainedNodes: !2)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{!"clang version 3.7.0 (trunk 226474) (llvm/trunk 226478)"}
!13 = !DILocation(line: 14, column: 3, scope: !4)
!14 = !DILocation(line: 15, column: 1, scope: !4)
!15 = !DILocation(line: 11, column: 3, scope: !7)
!16 = !DILocation(line: 12, column: 1, scope: !7)
!17 = !DILocation(line: 8, column: 3, scope: !8)
!18 = !DILocation(line: 9, column: 1, scope: !8)
!19 = !DILocation(line: 5, column: 3, scope: !9)
!20 = !DILocation(line: 6, column: 1, scope: !9)
