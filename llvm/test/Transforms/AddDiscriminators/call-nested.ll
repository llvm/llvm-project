; RUN: opt < %s -add-discriminators -S | FileCheck %s
; RUN: opt < %s -passes=add-discriminators -S | FileCheck %s

; Discriminator support for calls that are defined in one line:
; #1 int foo(int, int);
; #2 int bar();
; #3 int baz() {
; #4   return foo(bar(),
; #5              bar());
; #6 }

; Function Attrs: uwtable
define i32 @_Z3bazv() #0 !dbg !4 {
  %1 = call i32 @_Z3barv(), !dbg !11
; CHECK: %1 = call i32 @_Z3barv(), !dbg ![[CALL0:[0-9]+]]
  %2 = call i32 @_Z3barv(), !dbg !12
; CHECK: %2 = call i32 @_Z3barv(), !dbg ![[CALL1:[0-9]+]]
  %3 = call i32 @_Z3fooii(i32 %1, i32 %2), !dbg !13
; CHECK: %3 = call i32 @_Z3fooii(i32 %1, i32 %2), !dbg ![[CALL2:[0-9]+]]
  ret i32 %3, !dbg !14
}

declare i32 @_Z3fooii(i32, i32) #1

declare i32 @_Z3barv() #1

attributes #0 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 (trunk 266269)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.cc", directory: "")
!2 = !{}
!4 = distinct !DISubprogram(name: "baz", linkageName: "_Z3bazv", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang version 3.9.0 (trunk 266269)"}
!11 = !DILocation(line: 4, column: 14, scope: !4)
!12 = !DILocation(line: 5, column: 14, scope: !4)
!13 = !DILocation(line: 4, column: 10, scope: !4)
!14 = !DILocation(line: 4, column: 3, scope: !4)

; CHECK: ![[CALL2]] = !DILocation(line: 4, column: 10, scope: ![[CALL2BLOCK:[0-9]+]])
; CHECK: ![[CALL2BLOCK]] = !DILexicalBlockFile({{.*}} discriminator: 2)
