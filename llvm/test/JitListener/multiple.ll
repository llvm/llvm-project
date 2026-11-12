; Verify the behavior of the IntelJITEventListener.
; RUN: llvm-jitlistener %s | FileCheck %s

; This test was created using the following file:
;
;  1: int foo(int a) {
;  2:   return a;
;  3: }
;  4:
;  5: int bar(int a) {
;  6:   if (a == 0) {
;  7:     return 0;
;  8:   }
;  9:   return 100/a;
; 10: }
; 11: 
; 12: int fubar(int a) {
; 13:   switch (a) {
; 14:     case 0:
; 15:       return 10;
; 16:     case 1:
; 17:       return 20;
; 18:     default:
; 19:       return 30;
; 20:   }
; 21: }
;

; CHECK: Method load [1]: foo, Size = {{[0-9]+}}
; CHECK:   Line info @ {{[0-9]+}}: multiple.c, line {{[1,2]}}
; CHECK:   Line info @ {{[0-9]+}}: multiple.c, line {{[1,2]}}

; CHECK: Method load [2]: bar, Size = {{[0-9]+}}
; CHECK:   Line info @ {{[0-9]+}}: multiple.c, line {{[5,6,7,9]}}
; CHECK:   Line info @ {{[0-9]+}}: multiple.c, line {{[5,6,7,9]}}
; CHECK:   Line info @ {{[0-9]+}}: multiple.c, line {{[5,6,7,9]}}
; CHECK:   Line info @ {{[0-9]+}}: multiple.c, line {{[5,6,7,9]}}

; CHECK: Method load [3]: fubar, Size = {{[0-9]+}}
; CHECK:   Line info @ {{[0-9]+}}: multiple.c, line {{[12,13,15,17,19]}}
; CHECK:   Line info @ {{[0-9]+}}: multiple.c, line {{[12,13,15,17,19]}}
; CHECK:   Line info @ {{[0-9]+}}: multiple.c, line {{[12,13,15,17,19]}}
; CHECK:   Line info @ {{[0-9]+}}: multiple.c, line {{[12,13,15,17,19]}}
; CHECK:   Line info @ {{[0-9]+}}: multiple.c, line {{[12,13,15,17,19]}}

; CHECK: Method unload [1]
; CHECK: Method unload [2]
; CHECK: Method unload [3]

; ModuleID = 'multiple.c'

; Function Attrs: nounwind uwtable
define i32 @foo(i32 %a) #0 !dbg !4 {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !15, metadata !16), !dbg !17
  %0 = load i32, i32* %a.addr, align 4, !dbg !18
  ret i32 %0, !dbg !19
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind uwtable
define i32 @bar(i32 %a) #0 !dbg !9 {
entry:
  %retval = alloca i32, align 4
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !20, metadata !16), !dbg !21
  %0 = load i32, i32* %a.addr, align 4, !dbg !22
  %cmp = icmp eq i32 %0, 0, !dbg !22
  br i1 %cmp, label %if.then, label %if.end, !dbg !24

if.then:                                          ; preds = %entry
  store i32 0, i32* %retval, !dbg !25
  br label %return, !dbg !25

if.end:                                           ; preds = %entry
  %1 = load i32, i32* %a.addr, align 4, !dbg !27
  %div = sdiv i32 100, %1, !dbg !28
  store i32 %div, i32* %retval, !dbg !29
  br label %return, !dbg !29

return:                                           ; preds = %if.end, %if.then
  %2 = load i32, i32* %retval, !dbg !30
  ret i32 %2, !dbg !30
}

; Function Attrs: nounwind uwtable
define i32 @fubar(i32 %a) #0 !dbg !10 {
entry:
  %retval = alloca i32, align 4
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !31, metadata !16), !dbg !32
  %0 = load i32, i32* %a.addr, align 4, !dbg !33
  switch i32 %0, label %sw.default [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
  ], !dbg !34

sw.bb:                                            ; preds = %entry
  store i32 10, i32* %retval, !dbg !35
  br label %return, !dbg !35

sw.bb1:                                           ; preds = %entry
  store i32 20, i32* %retval, !dbg !37
  br label %return, !dbg !37

sw.default:                                       ; preds = %entry
  store i32 30, i32* %retval, !dbg !38
  br label %return, !dbg !38

return:                                           ; preds = %sw.default, %sw.bb1, %sw.bb
  %1 = load i32, i32* %retval, !dbg !39
  ret i32 %1, !dbg !39
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12, !13}
!llvm.ident = !{!14}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.6.0 (trunk)", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "multiple.c", directory: "F:\5Cusers\5Cakaylor\5Cllvm-s\5Cllvm\5Ctest\5CJitListener")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", line: 1, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "multiple.c", directory: "F:CusersCakaylorCllvm-sCllvmCtestCJitListener")
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = distinct !DISubprogram(name: "bar", line: 5, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 5, file: !1, scope: !5, type: !6, retainedNodes: !2)
!10 = distinct !DISubprogram(name: "fubar", line: 12, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 12, file: !1, scope: !5, type: !6, retainedNodes: !2)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"PIC Level", i32 2}
!14 = !{!"clang version 3.6.0 (trunk)"}
!15 = !DILocalVariable(name: "a", line: 1, arg: 1, scope: !4, file: !5, type: !8)
!16 = !DIExpression()
!17 = !DILocation(line: 1, column: 13, scope: !4)
!18 = !DILocation(line: 2, column: 10, scope: !4)
!19 = !DILocation(line: 2, column: 3, scope: !4)
!20 = !DILocalVariable(name: "a", line: 5, arg: 1, scope: !9, file: !5, type: !8)
!21 = !DILocation(line: 5, column: 13, scope: !9)
!22 = !DILocation(line: 6, column: 7, scope: !23)
!23 = distinct !DILexicalBlock(line: 6, column: 7, file: !1, scope: !9)
!24 = !DILocation(line: 6, column: 7, scope: !9)
!25 = !DILocation(line: 7, column: 5, scope: !26)
!26 = distinct !DILexicalBlock(line: 6, column: 15, file: !1, scope: !23)
!27 = !DILocation(line: 9, column: 14, scope: !9)
!28 = !DILocation(line: 9, column: 10, scope: !9)
!29 = !DILocation(line: 9, column: 3, scope: !9)
!30 = !DILocation(line: 10, column: 1, scope: !9)
!31 = !DILocalVariable(name: "a", line: 12, arg: 1, scope: !10, file: !5, type: !8)
!32 = !DILocation(line: 12, column: 15, scope: !10)
!33 = !DILocation(line: 13, column: 11, scope: !10)
!34 = !DILocation(line: 13, column: 3, scope: !10)
!35 = !DILocation(line: 15, column: 7, scope: !36)
!36 = distinct !DILexicalBlock(line: 13, column: 14, file: !1, scope: !10)
!37 = !DILocation(line: 17, column: 7, scope: !36)
!38 = !DILocation(line: 19, column: 7, scope: !36)
!39 = !DILocation(line: 21, column: 1, scope: !10)
