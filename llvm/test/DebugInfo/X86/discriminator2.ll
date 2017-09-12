; RUN: llc -mtriple=i386-unknown-unknown -mcpu=core2 %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-line %t | FileCheck %s
;
; Generated from:
;
; #1 void foo(int, int);
; #2 int bar();
; #3 void baz() {
; #4   foo/*discriminator 1*/(bar(),
; #5       bar());bar()/*discriminator 1*/;
; #6 }
;
; The intent is to test discriminator 1 generated for both line #4 and #5.
; The instruction sequence in the final binary is:
; line 4 discriminator 0
; line 5 discriminator 0
; line 4 discriminator 1
; line 5 discriminator 1
; We need to ensure that the discriminators for the last two instructions
; are both 1.

; Function Attrs: uwtable
define void @_Z3bazv() #0 !dbg !6 {
  %1 = call i32 @_Z3barv(), !dbg !9
  %2 = call i32 @_Z3barv(), !dbg !10
  call void @_Z3fooii(i32 %1, i32 %2), !dbg !11
  %3 = call i32 @_Z3barv(), !dbg !13
  ret void, !dbg !14
}

declare void @_Z3fooii(i32, i32) #1

declare i32 @_Z3barv() #1

attributes #0 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 (trunk 267219)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.cc", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 3.9.0 (trunk 267219)"}
!6 = distinct !DISubprogram(name: "baz", linkageName: "_Z3bazv", scope: !1, file: !1, line: 3, type: !7, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !DILocation(line: 4, column: 7, scope: !6)
!10 = !DILocation(line: 5, column: 14, scope: !6)
!11 = !DILocation(line: 4, column: 3, scope: !12)
!12 = !DILexicalBlockFile(scope: !6, file: !1, discriminator: 1)
!13 = !DILocation(line: 5, column: 21, scope: !12)
!14 = !DILocation(line: 6, column: 1, scope: !6)

; CHECK: Address            Line   Column File   ISA Discriminator Flags
; CHECK: ------------------ ------ ------ ------ --- ------------- -------------
; CHECK: {{.*}}      4      3      1   0             1  {{.*}}
; CHECK: {{.*}}      5     21      1   0             1  {{.*}}
