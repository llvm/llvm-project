; RUN: llc -mtriple=i386-unknown-unknown -mcpu=core2 %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-line %t | FileCheck %s
;
; Generated from:
;
; #1 void foo(int);
; #2 void baz(int i) {
; #3   if (i) {foo(i+1);/*discriminator 1*/}
; #4 }
;
; The intent is to test discriminator 1 generated for all instructions in
; the taken branch.

; Function Attrs: uwtable
define void @_Z3bazi(i32) #0 !dbg !6 {
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  call void @llvm.dbg.declare(metadata ptr %2, metadata !10, metadata !11), !dbg !12
  %3 = load i32, ptr %2, align 4, !dbg !13
  %4 = icmp ne i32 %3, 0, !dbg !13
  br i1 %4, label %5, label %8, !dbg !15

; <label>:5:                                      ; preds = %1
  %6 = load i32, ptr %2, align 4, !dbg !16
  %7 = add nsw i32 %6, 1, !dbg !19
  call void @_Z3fooi(i32 %7), !dbg !20
  br label %8, !dbg !21

; <label>:8:                                      ; preds = %5, %1
  ret void, !dbg !22
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @_Z3fooi(i32) #2

attributes #0 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 (trunk 267518)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.cc", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 3.9.0 (trunk 267518)"}
!6 = distinct !DISubprogram(name: "baz", linkageName: "_Z3bazi", scope: !1, file: !1, line: 2, type: !7, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9}
!9 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocalVariable(name: "i", arg: 1, scope: !6, file: !1, line: 2, type: !9)
!11 = !DIExpression()
!12 = !DILocation(line: 2, column: 14, scope: !6)
!13 = !DILocation(line: 3, column: 7, scope: !14)
!14 = distinct !DILexicalBlock(scope: !6, file: !1, line: 3, column: 7)
!15 = !DILocation(line: 3, column: 7, scope: !6)
!16 = !DILocation(line: 3, column: 15, scope: !17)
!17 = !DILexicalBlockFile(scope: !18, file: !1, discriminator: 1)
!18 = distinct !DILexicalBlock(scope: !14, file: !1, line: 3, column: 10)
!19 = !DILocation(line: 3, column: 16, scope: !17)
!20 = !DILocation(line: 3, column: 11, scope: !17)
!21 = !DILocation(line: 3, column: 21, scope: !17)
!22 = !DILocation(line: 4, column: 1, scope: !6)

; CHECK: Address            Line   Column File   ISA Discriminator OpIndex Flags
; CHECK: ------------------ ------ ------ ------ --- ------------- ------- -------------
; CHECK: {{.*}}                  3     15      1   0             1       0
; CHECK: {{.*}}                  3     16      1   0             1       0
; CHECK: {{.*}}                  3     11      1   0             1       0
