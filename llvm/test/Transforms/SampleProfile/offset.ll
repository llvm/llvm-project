; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/offset.prof | opt -passes='print<branch-prob>' -disable-output 2>&1 | FileCheck %s

; Original C++ code for this test case:
;
; a.cc:
; #1
; #2
; #3
; #4
; #5 int foo(int a) {
; #6 #include "a.b"
; #7}
;
; a.b:
; #1 if (a > 0) {
; #2   return 10;
; #3 } else {
; #4   return 20;
; #5 }

; Function Attrs: nounwind uwtable
define i32 @_Z3fooi(i32 %a) #0 !dbg !4 {
entry:
  %retval = alloca i32, align 4
  %a.addr = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %a.addr, metadata !11, metadata !12), !dbg !13
  %0 = load i32, ptr %a.addr, align 4, !dbg !14
  %cmp = icmp sgt i32 %0, 0, !dbg !18
  br i1 %cmp, label %if.then, label %if.else, !dbg !19
; CHECK: edge %entry -> %if.then probability is 0x0167ba82 / 0x80000000 = 1.10%
; CHECK: edge %entry -> %if.else probability is 0x7e98457e / 0x80000000 = 98.90% [HOT edge]

if.then:                                          ; preds = %entry
  store i32 10, ptr %retval, align 4, !dbg !20
  br label %return, !dbg !20

if.else:                                          ; preds = %entry
  store i32 20, ptr %retval, align 4, !dbg !22
  br label %return, !dbg !22

return:                                           ; preds = %if.else, %if.then
  %1 = load i32, ptr %retval, align 4, !dbg !24
  ret i32 %1, !dbg !24
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" "use-sample-profile" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 250750)", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "a.cc", directory: "/tmp")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !1, file: !1, line: 5, type: !5, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang version 3.8.0 (trunk 250750)"}
!11 = !DILocalVariable(name: "a", arg: 1, scope: !4, file: !1, line: 5, type: !7)
!12 = !DIExpression()
!13 = !DILocation(line: 5, column: 13, scope: !4)
!14 = !DILocation(line: 1, column: 5, scope: !15)
!15 = distinct !DILexicalBlock(scope: !17, file: !16, line: 1, column: 5)
!16 = !DIFile(filename: "./a.b", directory: "/tmp")
!17 = !DILexicalBlockFile(scope: !4, file: !16, discriminator: 0)
!18 = !DILocation(line: 1, column: 7, scope: !15)
!19 = !DILocation(line: 1, column: 5, scope: !17)
!20 = !DILocation(line: 2, column: 3, scope: !21)
!21 = distinct !DILexicalBlock(scope: !15, file: !16, line: 1, column: 12)
!22 = !DILocation(line: 4, column: 3, scope: !23)
!23 = distinct !DILexicalBlock(scope: !15, file: !16, line: 3, column: 8)
!24 = !DILocation(line: 7, column: 1, scope: !25)
!25 = !DILexicalBlockFile(scope: !4, file: !1, discriminator: 0)
