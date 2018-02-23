; RUN: llc -split-dwarf-file=foo.dwo -O0 < %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; CHECK-NOT: DW_TAG_subprogram

; IR generated from the following source:
; void f1();
; inline __attribute__((always_inline)) void f2() {
;   f1();
; }
; void f3() {
;   f2();
; }

; Function Attrs: uwtable
define void @_Z2f3v() #0 !dbg !5 {
entry:
  call void @_Z2f1v(), !dbg !8
  ret void, !dbg !11
}

declare void @_Z2f1v() #1

attributes #0 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}
!llvm.ident = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 4.0.0 (trunk 279571) (llvm/trunk 279602)", isOptimized: false, runtimeVersion: 0, splitDebugFilename: "fission-no-inlining.dwo", emissionKind: FullDebug, enums: !2, splitDebugInlining: false)
!1 = !DIFile(filename: "fission-no-inlining.cpp", directory: "/tmp/dbginfo")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{!"clang version 4.0.0 (trunk 279571) (llvm/trunk 279602)"}
!5 = distinct !DISubprogram(name: "f3", linkageName: "_Z2f3v", scope: !1, file: !1, line: 5, type: !6, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !DILocation(line: 3, column: 3, scope: !9, inlinedAt: !10)
!9 = distinct !DISubprogram(name: "f2", linkageName: "_Z2f2v", scope: !1, file: !1, line: 2, type: !6, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!10 = distinct !DILocation(line: 6, column: 3, scope: !5)
!11 = !DILocation(line: 7, column: 1, scope: !5)
