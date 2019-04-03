; RUN: opt < %s -argpromotion -S | FileCheck %s
; RUN: opt < %s -passes=argpromotion -S | FileCheck %s

declare void @sink(i32)

; CHECK: define internal void @test({{.*}} !dbg [[SP:![0-9]+]]
define internal void @test(i32** %X) !dbg !2 {
  %1 = load i32*, i32** %X, align 8
  %2 = load i32, i32* %1, align 8
  call void @sink(i32 %2)
  ret void
}

%struct.pair = type { i32, i32 }

; CHECK: define internal void @test_byval(i32 %{{.*}}, i32 %{{.*}})
define internal void @test_byval(%struct.pair* byval %P) {
  ret void
}

; CHECK-LABEL: define {{.*}} @caller(
define void @caller(i32** %Y, %struct.pair* %P) {
; CHECK:  load i32*, {{.*}} !dbg [[LOC_1:![0-9]+]]
; CHECK-NEXT:  load i32, {{.*}} !dbg [[LOC_1]]
; CHECK-NEXT: call void @test(i32 %{{.*}}), !dbg [[LOC_1]]
  call void @test(i32** %Y), !dbg !1

; CHECK: getelementptr %struct.pair, {{.*}} !dbg [[LOC_2:![0-9]+]]
; CHECK-NEXT: load i32, i32* {{.*}} !dbg [[LOC_2]]
; CHECK-NEXT: getelementptr %struct.pair, {{.*}} !dbg [[LOC_2]]
; CHECK-NEXT: load i32, i32* {{.*}} !dbg [[LOC_2]]
; CHECK-NEXT: call void @test_byval(i32 %{{.*}}, i32 %{{.*}}), !dbg [[LOC_2]]
  call void @test_byval(%struct.pair* %P), !dbg !6
  ret void
}

; CHECK: [[SP]] = distinct !DISubprogram(name: "test",
; CHECK: [[LOC_1]] = !DILocation(line: 8
; CHECK: [[LOC_2]] = !DILocation(line: 9

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !DILocation(line: 8, scope: !2)
!2 = distinct !DISubprogram(name: "test", file: !5, line: 3, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !3, scopeLine: 3, scope: null)
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: LineTablesOnly, file: !5)
!5 = !DIFile(filename: "test.c", directory: "")
!6 = !DILocation(line: 9, scope: !2)
