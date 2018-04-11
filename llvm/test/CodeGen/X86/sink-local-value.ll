; RUN: llc -fast-isel-sink-local-values -O0 < %s | FileCheck %s

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-linux-gnu"

; Try some simple cases that show how local value sinking improves line tables.

@sink_across = external global i32

declare void @simple_callee(i32, i32)

define void @simple() !dbg !5 {
  store i32 44, i32* @sink_across, !dbg !7
  call void @simple_callee(i32 13, i32 55), !dbg !8
  ret void, !dbg !9
}

; CHECK-LABEL: simple:
; CHECK-NOT: movl $13,
; CHECK: .loc 1 1 1 prologue_end
; CHECK: movl $44, sink_across
; CHECK: .loc 1 2 1
; CHECK: movl $13,
; CHECK: movl $55,
; CHECK: calll simple_callee

declare void @simple_reg_callee(i32 inreg, i32 inreg)

define void @simple_reg() !dbg !10 {
  store i32 44, i32* @sink_across, !dbg !11
  call void @simple_reg_callee(i32 inreg 13, i32 inreg 55), !dbg !12
  ret void, !dbg !13
}

; CHECK-LABEL: simple_reg:
; CHECK: .loc 1 4 1 prologue_end
; CHECK: movl $44, sink_across
; CHECK: .loc 1 5 1
; CHECK: movl $13,
; CHECK: movl $55,
; CHECK: calll simple_reg_callee

; There are two interesting cases where local values have no uses but are not
; dead: when the local value is directly used by a phi, and when the local
; value is used by a no-op cast instruction. In these cases, we get side tables
; referring to the local value vreg that we need to check.

define i8* @phi_const(i32 %c) !dbg !14 {
entry:
  %tobool = icmp eq i32 %c, 0, !dbg !20
  call void @llvm.dbg.value(metadata i1 %tobool, metadata !16, metadata !DIExpression()), !dbg !20
  br i1 %tobool, label %if.else, label %if.then, !dbg !21

if.then:                                          ; preds = %entry
  br label %if.end, !dbg !22

if.else:                                          ; preds = %entry
  br label %if.end, !dbg !23

if.end:                                           ; preds = %if.else, %if.then
  %r.0 = phi i8* [ inttoptr (i32 42 to i8*), %if.then ], [ inttoptr (i32 1 to i8*), %if.else ], !dbg !24
  call void @llvm.dbg.value(metadata i8* %r.0, metadata !18, metadata !DIExpression()), !dbg !24
  ret i8* %r.0, !dbg !25
}

; CHECK-LABEL: phi_const:
; CHECK:                                 # %entry
; CHECK: cmpl    $0,
; CHECK:                                 # %if.then
; CHECK: movl    $42,
; CHECK: jmp
; CHECK:                                 # %if.else
; CHECK: movl    $1,
; CHECK:                                 # %if.end

define i8* @phi_const_cast(i32 %c) !dbg !26 {
entry:
  %tobool = icmp eq i32 %c, 0, !dbg !32
  call void @llvm.dbg.value(metadata i1 %tobool, metadata !28, metadata !DIExpression()), !dbg !32
  br i1 %tobool, label %if.else, label %if.then, !dbg !33

if.then:                                          ; preds = %entry
  %v42 = inttoptr i32 42 to i8*, !dbg !34
  call void @llvm.dbg.value(metadata i8* %v42, metadata !29, metadata !DIExpression()), !dbg !34
  br label %if.end, !dbg !35

if.else:                                          ; preds = %entry
  %v1 = inttoptr i32 1 to i8*, !dbg !36
  call void @llvm.dbg.value(metadata i8* %v1, metadata !30, metadata !DIExpression()), !dbg !36
  br label %if.end, !dbg !37

if.end:                                           ; preds = %if.else, %if.then
  %r.0 = phi i8* [ %v42, %if.then ], [ %v1, %if.else ], !dbg !38
  call void @llvm.dbg.value(metadata i8* %r.0, metadata !31, metadata !DIExpression()), !dbg !38
  ret i8* %r.0, !dbg !39
}

; CHECK-LABEL: phi_const_cast:
; CHECK:                                 # %entry
; CHECK: cmpl    $0,
; CHECK:                                 # %if.then
; CHECK: movl    $42, %[[REG:[a-z]+]]
; CHECK: #DEBUG_VALUE: phi_const_cast:4 <- $[[REG]]
; CHECK: jmp
; CHECK:                                 # %if.else
; CHECK: movl    $1, %[[REG:[a-z]+]]
; CHECK: #DEBUG_VALUE: phi_const_cast:5 <- $[[REG]]
; CHECK:                                 # %if.end

declare void @may_throw() local_unnamed_addr #1

declare i32 @__gxx_personality_v0(...)

define i32 @invoke_phi() personality i32 (...)* @__gxx_personality_v0 {
entry:
  store i32 42, i32* @sink_across
  invoke void @may_throw()
          to label %try.cont unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          catch i8* null
  store i32 42, i32* @sink_across
  br label %try.cont

try.cont:                                         ; preds = %entry, %lpad
  %r.0 = phi i32 [ 13, %entry ], [ 55, %lpad ]
  ret i32 %r.0
}

; The constant materialization should be *after* the stores to sink_across, but
; before any EH_LABEL.

; CHECK-LABEL: invoke_phi:
; CHECK:         movl    $42, sink_across
; CHECK:         movl    $13, %{{[a-z]*}}
; CHECK: .Ltmp{{.*}}:
; CHECK:         calll   may_throw
; CHECK: .Ltmp{{.*}}:
; CHECK:         jmp     .LBB{{.*}}
; CHECK: .LBB{{.*}}:                                # %lpad
; CHECK:         movl    $42, sink_across
; CHECK:         movl    $55, %{{[a-z]*}}
; CHECK: .LBB{{.*}}:                                # %try.cont
; CHECK:         retl


; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!3, !4}
!llvm.module.flags = !{!52, !53}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "../llvm/test/CodeGen/X86/sink-local-value.ll", directory: "/")
!2 = !{}
!3 = !{i32 27}
!4 = !{i32 8}
!5 = distinct !DISubprogram(name: "simple", linkageName: "simple", scope: null, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !0, variables: !2)
!6 = !DISubroutineType(types: !2)
!7 = !DILocation(line: 1, column: 1, scope: !5)
!8 = !DILocation(line: 2, column: 1, scope: !5)
!9 = !DILocation(line: 3, column: 1, scope: !5)
!10 = distinct !DISubprogram(name: "simple_reg", linkageName: "simple_reg", scope: null, file: !1, line: 4, type: !6, isLocal: false, isDefinition: true, scopeLine: 4, isOptimized: true, unit: !0, variables: !2)
!11 = !DILocation(line: 4, column: 1, scope: !10)
!12 = !DILocation(line: 5, column: 1, scope: !10)
!13 = !DILocation(line: 6, column: 1, scope: !10)
!14 = distinct !DISubprogram(name: "phi_const", linkageName: "phi_const", scope: null, file: !1, line: 7, type: !6, isLocal: false, isDefinition: true, scopeLine: 7, isOptimized: true, unit: !0, variables: !15)
!15 = !{!16, !18}
!16 = !DILocalVariable(name: "1", scope: !14, file: !1, line: 7, type: !17)
!17 = !DIBasicType(name: "ty8", size: 8, encoding: DW_ATE_unsigned)
!18 = !DILocalVariable(name: "2", scope: !14, file: !1, line: 11, type: !19)
!19 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!20 = !DILocation(line: 7, column: 1, scope: !14)
!21 = !DILocation(line: 8, column: 1, scope: !14)
!22 = !DILocation(line: 9, column: 1, scope: !14)
!23 = !DILocation(line: 10, column: 1, scope: !14)
!24 = !DILocation(line: 11, column: 1, scope: !14)
!25 = !DILocation(line: 12, column: 1, scope: !14)
!26 = distinct !DISubprogram(name: "phi_const_cast", linkageName: "phi_const_cast", scope: null, file: !1, line: 13, type: !6, isLocal: false, isDefinition: true, scopeLine: 13, isOptimized: true, unit: !0, variables: !27)
!27 = !{!28, !29, !30, !31}
!28 = !DILocalVariable(name: "3", scope: !26, file: !1, line: 13, type: !17)
!29 = !DILocalVariable(name: "4", scope: !26, file: !1, line: 15, type: !19)
!30 = !DILocalVariable(name: "5", scope: !26, file: !1, line: 17, type: !19)
!31 = !DILocalVariable(name: "6", scope: !26, file: !1, line: 19, type: !19)
!32 = !DILocation(line: 13, column: 1, scope: !26)
!33 = !DILocation(line: 14, column: 1, scope: !26)
!34 = !DILocation(line: 15, column: 1, scope: !26)
!35 = !DILocation(line: 16, column: 1, scope: !26)
!36 = !DILocation(line: 17, column: 1, scope: !26)
!37 = !DILocation(line: 18, column: 1, scope: !26)
!38 = !DILocation(line: 19, column: 1, scope: !26)
!39 = !DILocation(line: 20, column: 1, scope: !26)
!40 = distinct !DISubprogram(name: "invoke_phi", linkageName: "invoke_phi", scope: null, file: !1, line: 21, type: !6, isLocal: false, isDefinition: true, scopeLine: 21, isOptimized: true, unit: !0, variables: !41)
!41 = !{!42, !44}
!42 = !DILocalVariable(name: "7", scope: !40, file: !1, line: 23, type: !43)
!43 = !DIBasicType(name: "ty64", size: 64, encoding: DW_ATE_unsigned)
!44 = !DILocalVariable(name: "8", scope: !40, file: !1, line: 26, type: !19)
!45 = !DILocation(line: 21, column: 1, scope: !40)
!46 = !DILocation(line: 22, column: 1, scope: !40)
!47 = !DILocation(line: 23, column: 1, scope: !40)
!48 = !DILocation(line: 24, column: 1, scope: !40)
!49 = !DILocation(line: 25, column: 1, scope: !40)
!50 = !DILocation(line: 26, column: 1, scope: !40)
!51 = !DILocation(line: 27, column: 1, scope: !40)
!52 = !{i32 2, !"Dwarf Version", i32 4}
!53 = !{i32 2, !"Debug Info Version", i32 3}
