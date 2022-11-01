; RUN: opt -S -o - -structurizecfg %s | FileCheck %s

define void @if_then_else(i32 addrspace(1)* %out, i1 %arg) !dbg !7 {
; CHECK: @if_then_else(
; CHECK:  entry:
; CHECK:    br i1 {{.*}}, label %if.else, label %Flow, !dbg [[ITE_ENTRY_DL:![0-9]+]]
; CHECK:  Flow:
; CHECK:    br i1 {{.*}}, label %if.then, label %exit, !dbg [[ITE_ENTRY_DL]]
; CHECK:  if.then:
; CHECK:    br label %exit, !dbg [[ITE_IFTHEN_DL:![0-9]+]]
; CHECK:  if.else:
; CHECK:    br label %Flow, !dbg [[ITE_IFELSE_DL:![0-9]+]]
; CHECK:  exit:
;
entry:
  br i1 %arg, label %if.then, label %if.else, !dbg !8

if.then:
  store i32 0, i32 addrspace(1)* %out, !dbg !9
  br label %exit, !dbg !10

if.else:
  store i32 1, i32 addrspace(1)* %out, !dbg !11
  br label %exit, !dbg !12

exit:
  ret void, !dbg !13
}

define void @while_loop(i32 addrspace(1)* %out) !dbg !14 {
; CHECK: @while_loop(
; CHECK:  entry:
; CHECK:    br label %while.header, !dbg [[WHILE_ENTRY_DL:![0-9]+]]
; CHECK:  while.header:
; CHECK:    br i1 {{.*}}, label %while.body, label %Flow, !dbg [[WHILE_HEADER_DL:![0-9]+]]
; CHECK:  while.body:
; CHECK:    br label %Flow, !dbg [[WHILE_BODY_DL:![0-9]+]]
; CHECK:  Flow:
; CHECK:    br i1 {{.*}}, label %exit, label %while.header, !dbg [[WHILE_HEADER_DL]]
; CHECK:  exit:
;
entry:
  br label %while.header, !dbg !15

while.header:
  %cond = call i1 @loop_condition(), !dbg !16
  br i1 %cond, label %while.body, label %exit, !dbg !17

while.body:
  store i32 1, i32 addrspace(1)* %out, !dbg !18
  br label %while.header, !dbg !19

exit:
  ret void, !dbg !20
}

define void @while_multiple_exits(i32 addrspace(1)* %out) !dbg !21 {
; CHECK: @while_multiple_exits(
; CHECK:  entry:
; CHECK:    br label %while.header, !dbg [[WHILEME_ENTRY_DL:![0-9]+]]
; CHECK:  while.header:
; CHECK:    br i1 {{.*}}, label %while.exiting, label %Flow, !dbg [[WHILEME_HEADER_DL:![0-9]+]]
; CHECK:  while.exiting:
; CHECK:    br label %Flow, !dbg [[WHILEME_EXITING_DL:![0-9]+]]
; CHECK:  Flow:
; CHECK:    br i1 {{.*}}, label %exit, label %while.header, !dbg [[WHILEME_HEADER_DL]] 
; CHECK:  exit:
;
entry:
  br label %while.header, !dbg !22

while.header:
  %cond0 = call i1 @loop_condition(), !dbg !23
  br i1 %cond0, label %while.exiting, label %exit, !dbg !24

while.exiting:
  %cond1 = call i1 @loop_condition(), !dbg !25
  br i1 %cond1, label %while.header, label %exit, !dbg !26

exit:
  ret void, !dbg !27
}

define void @nested_if_then_else(i32 addrspace(1)* %out, i1 %a, i1 %b) !dbg !28 {
; CHECK: @nested_if_then_else(
; CHECK:  entry:
; CHECK:    br i1 {{.*}}, label %if.else, label %Flow4, !dbg [[NESTED_ENTRY_DL:![0-9]+]]
; CHECK:  Flow4:
; CHECK:    br i1 {{.*}}, label %if.then, label %exit, !dbg [[NESTED_ENTRY_DL]]
; CHECK:  if.then:
; CHECK:    br i1 {{.*}}, label %if.then.else, label %Flow2, !dbg [[NESTED_IFTHEN_DL:![0-9]+]]
; CHECK:  Flow2:
; CHECK:    br i1 {{.*}}, label %if.then.then, label %Flow3, !dbg [[NESTED_IFTHEN_DL]]
; CHECK:  if.then.then:
; CHECK:    br label %Flow3, !dbg [[NESTED_IFTHENTHEN_DL:![0-9]+]]
; CHECK:  if.then.else:
; CHECK:    br label %Flow2, !dbg [[NESTED_IFTHENELSE_DL:![0-9]+]]
; CHECK:  if.else:
; CHECK:    br i1 {{.*}}, label %if.else.else, label %Flow, !dbg [[NESTED_IFELSE_DL:![0-9]+]]
; CHECK:  Flow:
; CHECK:    br i1 {{.*}}, label %if.else.then, label %Flow1, !dbg [[NESTED_IFELSE_DL]]
; CHECK:  if.else.then:
; CHECK:    br label %Flow1, !dbg [[NESTED_IFELSETHEN_DL:![0-9]+]]
; CHECK:  if.else.else:
; CHECK:    br label %Flow, !dbg [[NESTED_IFELSEELSE_DL:![0-9]+]]
; CHECK:  Flow1:
; CHECK:    br label %Flow4, !dbg [[NESTED_IFELSE_DL]]
; CHECK:  Flow3:
; CHECK:    br label %exit, !dbg [[NESTED_IFTHEN_DL]]
; CHECK:  exit:
;
entry:
  br i1 %a, label %if.then, label %if.else, !dbg !29

if.then:
  br i1 %b, label %if.then.then, label %if.then.else, !dbg !30

if.then.then:
  store i32 0, i32 addrspace(1)* %out, !dbg !31
  br label %exit, !dbg !32

if.then.else:
  store i32 1, i32 addrspace(1)* %out, !dbg !33
  br label %exit, !dbg !34

if.else:
  br i1 %b, label %if.else.then, label %if.else.else, !dbg !35

if.else.then:
  store i32 2, i32 addrspace(1)* %out, !dbg !36
  br label %exit, !dbg !37

if.else.else:
  store i32 3, i32 addrspace(1)* %out, !dbg !38
  br label %exit, !dbg !39

exit:
  ret void, !dbg !40
}

declare i1 @loop_condition()

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!4, !5}

; CHECK: [[ITE_ENTRY_DL]] = !DILocation(line: 2
; CHECK: [[ITE_IFTHEN_DL]] = !DILocation(line: 4
; CHECK: [[ITE_IFELSE_DL]] = !DILocation(line: 6
; CHECK: [[WHILE_ENTRY_DL]] = !DILocation(line: 2
; CHECK: [[WHILE_HEADER_DL]] = !DILocation(line: 4
; CHECK: [[WHILE_BODY_DL]] = !DILocation(line: 6
; CHECK: [[WHILEME_ENTRY_DL]] = !DILocation(line: 2
; CHECK: [[WHILEME_HEADER_DL]] = !DILocation(line: 4
; CHECK: [[WHILEME_EXITING_DL]] = !DILocation(line: 6
; CHECK: [[NESTED_ENTRY_DL]] = !DILocation(line: 2
; CHECK: [[NESTED_IFTHEN_DL]] = !DILocation(line: 3
; CHECK: [[NESTED_IFTHENTHEN_DL]] = !DILocation(line: 5
; CHECK: [[NESTED_IFTHENELSE_DL]] = !DILocation(line: 7
; CHECK: [[NESTED_IFELSE_DL]] = !DILocation(line: 8
; CHECK: [[NESTED_IFELSETHEN_DL]] = !DILocation(line: 10 
; CHECK: [[NESTED_IFELSEELSE_DL]] = !DILocation(line: 12

!0 = !{}
!1 = !DIFile(filename: "dummy.ll", directory: "/some/random/directory")
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !0)
!4 = !{i32 2, !"Dwarf Version", i32 5}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !DISubroutineType(types: !0)
!7 = distinct !DISubprogram(name: "dummy", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !2, retainedNodes: !0)
!8 = !DILocation(line: 2, scope: !7)
!9 = !DILocation(line: 3, scope: !7)
!10 = !DILocation(line: 4, scope: !7)
!11 = !DILocation(line: 5, scope: !7)
!12 = !DILocation(line: 6, scope: !7)
!13 = !DILocation(line: 7, scope: !7)
!14 = distinct !DISubprogram(name: "dummy", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !2, retainedNodes: !0)
!15 = !DILocation(line: 2, scope: !14)
!16 = !DILocation(line: 3, scope: !14)
!17 = !DILocation(line: 4, scope: !14)
!18 = !DILocation(line: 5, scope: !14)
!19 = !DILocation(line: 6, scope: !14)
!20 = !DILocation(line: 7, scope: !14)
!21 = distinct !DISubprogram(name: "dummy", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !2, retainedNodes: !0)
!22 = !DILocation(line: 2, scope: !21)
!23 = !DILocation(line: 3, scope: !21)
!24 = !DILocation(line: 4, scope: !21)
!25 = !DILocation(line: 5, scope: !21)
!26 = !DILocation(line: 6, scope: !21)
!27 = !DILocation(line: 7, scope: !21)
!28 = distinct !DISubprogram(name: "dummy", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !2, retainedNodes: !0)
!29 = !DILocation(line: 2, scope: !28)
!30 = !DILocation(line: 3, scope: !28)
!31 = !DILocation(line: 4, scope: !28)
!32 = !DILocation(line: 5, scope: !28)
!33 = !DILocation(line: 6, scope: !28)
!34 = !DILocation(line: 7, scope: !28)
!35 = !DILocation(line: 8, scope: !28)
!36 = !DILocation(line: 9, scope: !28)
!37 = !DILocation(line: 10, scope: !28)
!38 = !DILocation(line: 11, scope: !28)
!39 = !DILocation(line: 12, scope: !28)
!40 = !DILocation(line: 13, scope: !28)
