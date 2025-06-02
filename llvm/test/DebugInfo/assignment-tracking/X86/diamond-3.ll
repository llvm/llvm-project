; RUN: llc %s -stop-after=finalize-isel -o - \
; RUN: | FileCheck %s --implicit-check-not=DBG_

;; Hand written to test scenario we can definitely run into in the wild. This
;; file name includes "diamond" because the idea is that we lose (while
;; optimizing) one of the diamond branches which was empty except for a debug
;; intrinsic. In this case, the debug intrinsic linked to the common-and-sunk
;; store now in if.end. So we've got this:
;;
;; entry:          ; -> br if.then, if.end
;;    mem(a) = !19
;;    dbg(a) = !21 ; dbg and mem disagree, don't use mem loc.
;; if.then:        ; -> br if.end
;;    dbg(a) = !20
;; if.end:
;;    mem(a) = !20 ; two preds disagree that !20 is the last assignment, don't
;;                 ; use mem loc.
;;    ; This feels highly unfortunate, and highlights the need to reinstate the
;;    ; memory location at call sites leaking the address (in an ideal world,
;;    ; the memory location would always be in use at that point and so this
;;    ; wouldn't be necessary).
;;    esc(a)       ; force the memory location

;; In real world examples this is caused by InstCombine sinking common code
;; followed by SimplifyCFG deleting empty-except-for-dbg blocks.

; CHECK-DAG: ![[A:[0-9]+]] = !DILocalVariable(name: "a",
; CHECK-LABEL: bb.0.entry:
; CHECK:         DBG_VALUE $edi, $noreg, ![[A]], !DIExpression()
; CHECK-LABEL: bb.1.if.then:
; CHECK:         DBG_VALUE 0, $noreg, ![[A]], !DIExpression()

;; === TODO / WISHLIST ===
; LEBAL-KCEHC: bb.2.if.end:
; KCEHC:         CALL64pcrel32 target-flags(x86-plt) @es
; KCEHC:         DBG_VALUE %stack.0.a.addr, $noreg, ![[A]], !DIExpression(DW_OP_deref)

target triple = "x86_64-unknown-linux-gnu"

@g = dso_local local_unnamed_addr global ptr null, align 8, !dbg !0

define dso_local noundef i32 @_Z1fiii(i32 noundef %a, i32 noundef %b, i32 noundef %c) local_unnamed_addr #0 !dbg !12 {
entry:
  %a.addr = alloca i32, align 4, !DIAssignID !19
  call void @llvm.dbg.assign(metadata i1 undef, metadata !16, metadata !DIExpression(), metadata !19, metadata ptr %a.addr, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.assign(metadata i32 %a, metadata !16, metadata !DIExpression(), metadata !21, metadata ptr %a.addr, metadata !DIExpression()), !dbg !20
  %tobool.not = icmp eq i32 %c, 0
  br i1 %tobool.not, label %if.then, label %if.end

if.then:
  call void @e()
  call void @llvm.dbg.assign(metadata i32 0, metadata !16, metadata !DIExpression(), metadata !22, metadata ptr %a.addr, metadata !DIExpression()), !dbg !20
  br label %if.end

if.end:                                           ; preds = %do.body
  store i32 0, ptr %a.addr, align 4, !DIAssignID !22
  call void @es(ptr %a.addr)
  ret i32 0
}

declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata) #2
declare void @e()
declare void @es(ptr)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10, !1000}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 4, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.cpp", directory: "/")
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 7, !"Dwarf Version", i32 5}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 7, !"uwtable", i32 1}
!11 = !{!"clang version 14.0.0"}
!12 = distinct !DISubprogram(name: "f", linkageName: "_Z1fiii", scope: !3, file: !3, line: 5, type: !13, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !15)
!13 = !DISubroutineType(types: !14)
!14 = !{!6, !6, !6, !6}
!15 = !{!16, !17, !18}
!16 = !DILocalVariable(name: "a", arg: 1, scope: !12, file: !3, line: 5, type: !6)
!17 = !DILocalVariable(name: "b", arg: 2, scope: !12, file: !3, line: 5, type: !6)
!18 = !DILocalVariable(name: "c", arg: 3, scope: !12, file: !3, line: 5, type: !6)
!19 = distinct !DIAssignID()
!20 = !DILocation(line: 0, scope: !12)
!21 = distinct !DIAssignID()
!22 = distinct !DIAssignID()
!23 = distinct !DIAssignID()
!28 = distinct !DIAssignID()
!29 = !DILocation(line: 6, column: 3, scope: !12)
!30 = !DILocation(line: 8, column: 7, scope: !31)
!31 = distinct !DILexicalBlock(scope: !12, file: !3, line: 6, column: 6)
!32 = distinct !DIAssignID()
!33 = !DILocation(line: 10, column: 5, scope: !31)
!34 = !DILocation(line: 11, column: 12, scope: !12)
!35 = !DILocation(line: 11, column: 3, scope: !31)
!36 = distinct !{!36, !29, !37, !38}
!37 = !DILocation(line: 11, column: 15, scope: !12)
!38 = !{!"llvm.loop.mustprogress"}
!39 = !DILocation(line: 12, column: 3, scope: !12)
!40 = !DILocation(line: 13, column: 10, scope: !12)
!41 = !DILocation(line: 13, column: 12, scope: !12)
!42 = !DILocation(line: 13, column: 3, scope: !12)
!43 = !DISubprogram(name: "e", linkageName: "_Z1ev", scope: !3, file: !3, line: 2, type: !44, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !46)
!44 = !DISubroutineType(types: !45)
!45 = !{null}
!46 = !{}
!47 = !DISubprogram(name: "d", linkageName: "_Z1dv", scope: !3, file: !3, line: 1, type: !48, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !46)
!48 = !DISubroutineType(types: !49)
!49 = !{!6}
!50 = !DISubprogram(name: "es", linkageName: "_Z2esPi", scope: !3, file: !3, line: 3, type: !51, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !46)
!51 = !DISubroutineType(types: !52)
!52 = !{null, !5}
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
