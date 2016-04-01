; RUN: opt < %s -add-discriminators -S | FileCheck %s

; Discriminator support for calls that are defined in one line:
; #1 void bar();
; #2
; #3 void foo() {
; #4  bar();bar()/*discriminator 1*/;bar()/*discriminator 2*/;
; #5 }

; Function Attrs: uwtable
define void @_Z3foov() #0 !dbg !4 {
  call void @_Z3barv(), !dbg !10
; CHECK:  call void @_Z3barv(), !dbg ![[CALL0:[0-9]+]]
  call void @_Z3barv(), !dbg !11
; CHECK:  call void @_Z3barv(), !dbg ![[CALL1:[0-9]+]]
  call void @_Z3barv(), !dbg !12
; CHECK:  call void @_Z3barv(), !dbg ![[CALL2:[0-9]+]]
  ret void, !dbg !13
}

declare void @_Z3barv() #1

attributes #0 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

; We should be able to add discriminators even in the absence of llvm.dbg.cu.
; When using sample profiles, the front end will generate line tables but it
; does not generate llvm.dbg.cu to prevent codegen from emitting debug info
; to the final binary.
; !llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 250915) (llvm/trunk 251830)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, subprograms: !3)
!1 = !DIFile(filename: "c.cc", directory: "/tmp")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, variables: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang version 3.8.0 (trunk 250915) (llvm/trunk 251830)"}
!10 = !DILocation(line: 4, column: 3, scope: !4)
!11 = !DILocation(line: 4, column: 9, scope: !4)
!12 = !DILocation(line: 4, column: 15, scope: !4)
!13 = !DILocation(line: 5, column: 1, scope: !4)

; CHECK: ![[CALL1]] = !DILocation(line: 4, column: 9, scope: ![[CALL1BLOCK:[0-9]+]])
; CHECK: ![[CALL1BLOCK]] = !DILexicalBlockFile({{.*}} discriminator: 1)
; CHECK: ![[CALL2]] = !DILocation(line: 4, column: 15, scope: ![[CALL2BLOCK:[0-9]+]])
; CHECK: ![[CALL2BLOCK]] = !DILexicalBlockFile({{.*}} discriminator: 2)
