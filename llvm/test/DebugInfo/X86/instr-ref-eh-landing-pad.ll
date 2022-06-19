; RUN: llc %s -stop-before=finalize-isel -o -\
; RUN:     -experimental-debug-variable-locations \
; RUN:   | FileCheck %s
;
; Test that instruction referencing variable locations can cope with exception
; landing pads. Variable locations can be derived from the ABI-defined arguments
; to landing-pad blocks, which should be treated much like argument locations.
; This gloriously simple piece of C++:
;
;    void a() try { a(); } catch (int *&) { }
;
; Produces the code below. A nameless variable is attached to the landing-pad
; record %0 (via %3), which salvages back to the entry to the landing pad. We
; should place a DBG_PHI at that point. Instead of crashing.
;
; CHECK-LABEL: bb.1.lpad (landing-pad):
; CHECK:       DBG_PHI $rax, 1
; CHECK-NEXT:  EH_LABEL

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZTIPi = external dso_local constant i8*

define dso_local void @_Z1av() local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) !dbg !8 {
entry:
  invoke void @_Z1av()
          to label %try.cont unwind label %lpad, !dbg !17

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIPi to i8*), !dbg !19
  %1 = extractvalue { i8*, i32 } %0, 1, !dbg !19
  %2 = tail call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIPi to i8*)) #3, !dbg !20
  %matches = icmp eq i32 %1, %2, !dbg !20
  br i1 %matches, label %catch, label %eh.resume, !dbg !20

catch:                                            ; preds = %lpad
  %3 = extractvalue { i8*, i32 } %0, 0, !dbg !19
  %4 = tail call i8* @__cxa_begin_catch(i8* %3) #3, !dbg !20
  call void @llvm.dbg.value(metadata i8* %3, metadata !13, metadata !DIExpression(DW_OP_plus_uconst, 32, DW_OP_stack_value)), !dbg !21
  tail call void @__cxa_end_catch() #3, !dbg !22
  br label %try.cont, !dbg !22

try.cont:                                         ; preds = %entry, %catch
  ret void, !dbg !24

eh.resume:                                        ; preds = %lpad
  resume { i8*, i32 } %0, !dbg !20
}

declare dso_local i32 @__gxx_personality_v0(...)

; Function Attrs: nofree nosync nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #1

declare dso_local i8* @__cxa_begin_catch(i8*) local_unnamed_addr

declare dso_local void @__cxa_end_catch() local_unnamed_addr

; Function Attrs: mustprogress nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: ".")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"uwtable", i32 1}
!7 = !{!"clang"}
!8 = distinct !DISubprogram(name: "a", linkageName: "_Z1av", scope: !9, file: !9, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!9 = !DIFile(filename: "test.cpp", directory: "")
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !{!13}
!13 = !DILocalVariable(scope: !8, file: !9, line: 1, type: !14)
!14 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !15, size: 64)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !DILocation(line: 1, column: 16, scope: !18)
!18 = distinct !DILexicalBlock(scope: !8, file: !9, line: 1, column: 14)
!19 = !DILocation(line: 2, column: 1, scope: !18)
!20 = !DILocation(line: 1, column: 21, scope: !18)
!21 = !DILocation(line: 0, scope: !8)
!22 = !DILocation(line: 2, column: 1, scope: !23)
!23 = distinct !DILexicalBlock(scope: !8, file: !9, line: 1, column: 38)
!24 = !DILocation(line: 2, column: 1, scope: !8)
