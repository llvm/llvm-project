; Check that LoopSimplify creates debug locations in synthesized basic blocks.
; RUN: opt -loop-simplify %s -S -o - | FileCheck %s

%union.anon = type { i32 }
%"Length" = type <{ %union.anon, i8, i8, i8, i8 }>
declare void @bar(%"Length"*) #3
@catchtypeinfo = external unnamed_addr constant { i8*, i8*, i8* }
declare i32 @__gxx_personality_v0(...)
declare void @f1()
declare void @f2()
declare void @f3()

; CHECK-LABEL: @foo
; CHECK:       for.body.preheader:
; CHECK-NEXT:    br label %for.body, !dbg [[PREHEADER_LOC:![0-9]+]]
; CHECK:       for.end.loopexit:
; CHECK-NEXT:    br label %for.end, !dbg [[LOOPEXIT_LOC:![0-9]+]]

define linkonce_odr hidden void @foo(%"Length"* %begin, %"Length"* %end) nounwind ssp uwtable align 2 !dbg !6 {
entry:
  %cmp.4 = icmp eq %"Length"* %begin, %end, !dbg !7
  br i1 %cmp.4, label %for.end, label %for.body, !dbg !8

for.body:                                         ; preds = %entry, %length.exit
  %begin.sink5 = phi %"Length"* [ %incdec.ptr, %length.exit ], [ %begin, %entry ]
  tail call void @llvm.dbg.value(metadata %"Length"* %begin.sink5, metadata !15, metadata !16), !dbg !17
  %m_type.i.i.i = getelementptr inbounds %"Length", %"Length"* %begin.sink5, i64 0, i32 2, !dbg !9
  %0 = load i8, i8* %m_type.i.i.i, align 1, !dbg !9
  %cmp.i.i = icmp eq i8 %0, 9, !dbg !7
  br i1 %cmp.i.i, label %if.then.i, label %length.exit, !dbg !8

if.then.i:                                        ; preds = %for.body
  tail call void @bar(%"Length"* %begin.sink5) #7, !dbg !10
  br label %length.exit, !dbg !10

length.exit:                        ; preds = %for.body, %if.then.i
  %incdec.ptr = getelementptr inbounds %"Length", %"Length"* %begin.sink5, i64 1, !dbg !11
  %cmp = icmp eq %"Length"* %incdec.ptr, %end, !dbg !7
  br i1 %cmp, label %for.end, label %for.body, !dbg !8

for.end:                                          ; preds = %length.exit, %entry
  ret void, !dbg !12
}

; CHECK-LABEL: @with_landingpad
; CHECK: catch.preheader:
; CHECK:   br label %catch, !dbg [[LPAD_PREHEADER_LOC:![0-9]+]]
; CHECK: catch.preheader.split-lp:
; CHECK:   br label %catch, !dbg [[LPAD_PREHEADER_LOC]]

define void @with_landingpad() uwtable ssp personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @f1() to label %try.cont19 unwind label %catch, !dbg !13

catch:                                            ; preds = %if.else, %entry
  %0 = landingpad { i8*, i32 }
          catch i8* bitcast ({ i8*, i8*, i8* }* @catchtypeinfo to i8*), !dbg !13
  invoke void @f3() to label %if.else unwind label %eh.resume, !dbg !13

if.else:                                          ; preds = %catch
  invoke void @f2() to label %try.cont19 unwind label %catch, !dbg !13

try.cont19:                                       ; preds = %if.else, %entry
  ret void, !dbg !13

eh.resume:                                        ; preds = %catch
  %1 = landingpad { i8*, i32 }
          cleanup catch i8* bitcast ({ i8*, i8*, i8* }* @catchtypeinfo to i8*), !dbg !13
  resume { i8*, i32 } undef, !dbg !13
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata)

; CHECK-DAG: [[PREHEADER_LOC]] = !DILocation(line: 73, column: 13, scope: !{{[0-9]+}})
; CHECK-DAG: [[LOOPEXIT_LOC]] = !DILocation(line: 75, column: 9, scope: !{{[0-9]+}})
; CHECK-DAG: [[LPAD_PREHEADER_LOC]] = !DILocation(line: 85, column: 1, scope: !{{[0-9]+}})

!llvm.module.flags = !{!0, !1, !2}
!llvm.dbg.cu = !{!14}
!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{i32 1, !"PIC Level", i32 2}

!3 = !{}
!4 = !DISubroutineType(types: !3)
!5 = !DIFile(filename: "Vector.h", directory: "/tmp")
!6 = distinct !DISubprogram(name: "destruct", scope: !5, file: !5, line: 71, type: !4, isLocal: false, isDefinition: true, scopeLine: 72, flags: DIFlagPrototyped, isOptimized: false, unit: !14, retainedNodes: !3)
!7 = !DILocation(line: 73, column: 38, scope: !6)
!8 = !DILocation(line: 73, column: 13, scope: !6)
!9 = !DILocation(line: 73, column: 27, scope: !6)
!10 = !DILocation(line: 74, column: 17, scope: !6)
!11 = !DILocation(line: 73, column: 46, scope: !6)
!12 = !DILocation(line: 75, column: 9, scope: !6)
!13 = !DILocation(line: 85, column: 1, scope: !6)
!14 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang",
                             file: !5,
                             isOptimized: true, flags: "-O2",
                             splitDebugFilename: "abc.debug", emissionKind: 2)
!15 = !DILocalVariable(name: "begin", arg: 1, scope: !6, file: !5, line: 71)
!16 = !DIExpression()
!17 = !DILocation(line: 71, column: 32, scope: !6)
