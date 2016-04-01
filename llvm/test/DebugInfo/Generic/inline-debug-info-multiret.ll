; RUN: opt -inline -S < %s | FileCheck %s
;
; A hand-edited version of inline-debug-info.ll to test inlining of a
; function with multiple returns.
;
; Make sure the branch instructions created during inlining has a debug location,
; so the range of the inlined function is correct.
; CHECK: br label %_Z4testi.exit, !dbg ![[MD:[0-9]+]]
; CHECK: br label %_Z4testi.exit, !dbg ![[MD]]
; CHECK: br label %invoke.cont, !dbg ![[MD]]
; The branch instruction has the source location of line 9 and its inlined location
; has the source location of line 14.
; CHECK: ![[INL:[0-9]+]] = distinct !DILocation(line: 14, scope: {{.*}})
; CHECK: ![[MD]] = !DILocation(line: 9, scope: {{.*}}, inlinedAt: ![[INL]])

; ModuleID = 'test.cpp'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin12.0.0"

@_ZTIi = external constant i8*
@global_var = external global i32

; copy of above function with multiple returns
define i32 @_Z4testi(i32 %k)  !dbg !4 {
entry:
  %retval = alloca i32, align 4
  %k.addr = alloca i32, align 4
  %k2 = alloca i32, align 4
  store i32 %k, i32* %k.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %k.addr, metadata !13, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32* %k2, metadata !15, metadata !DIExpression()), !dbg !16
  %0 = load i32, i32* %k.addr, align 4, !dbg !16
  %call = call i32 @_Z8test_exti(i32 %0), !dbg !16
  store i32 %call, i32* %k2, align 4, !dbg !16
  %1 = load i32, i32* %k2, align 4, !dbg !17
  %cmp = icmp sgt i32 %1, 100, !dbg !17
  br i1 %cmp, label %if.then, label %if.end, !dbg !17

if.then:                                          ; preds = %entry
  %2 = load i32, i32* %k2, align 4, !dbg !18
  store i32 %2, i32* %retval, !dbg !18
  br label %return, !dbg !18

if.end:                                           ; preds = %entry
  store i32 0, i32* %retval, !dbg !19
  %3 = load i32, i32* %retval, !dbg !20                ; hand-edited
  ret i32 %3, !dbg !20                            ; hand-edited

return:                                           ; preds = %if.end, %if.then
  %4 = load i32, i32* %retval, !dbg !20
  ret i32 %4, !dbg !20
}


; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare i32 @_Z8test_exti(i32)

define i32 @_Z5test2v() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) !dbg !10 {
entry:
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  %e = alloca i32, align 4
  %0 = load i32, i32* @global_var, align 4, !dbg !21
  %call = invoke i32 @_Z4testi(i32 %0)
          to label %invoke.cont unwind label %lpad, !dbg !21

invoke.cont:                                      ; preds = %entry
  br label %try.cont, !dbg !23

lpad:                                             ; preds = %entry
  %1 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*), !dbg !21
  %2 = extractvalue { i8*, i32 } %1, 0, !dbg !21
  store i8* %2, i8** %exn.slot, !dbg !21
  %3 = extractvalue { i8*, i32 } %1, 1, !dbg !21
  store i32 %3, i32* %ehselector.slot, !dbg !21
  br label %catch.dispatch, !dbg !21

catch.dispatch:                                   ; preds = %lpad
  %sel = load i32, i32* %ehselector.slot, !dbg !23
  %4 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #2, !dbg !23
  %matches = icmp eq i32 %sel, %4, !dbg !23
  br i1 %matches, label %catch, label %eh.resume, !dbg !23

catch:                                            ; preds = %catch.dispatch
  call void @llvm.dbg.declare(metadata i32* %e, metadata !24, metadata !DIExpression()), !dbg !25
  %exn = load i8*, i8** %exn.slot, !dbg !23
  %5 = call i8* @__cxa_begin_catch(i8* %exn) #2, !dbg !23
  %6 = bitcast i8* %5 to i32*, !dbg !23
  %7 = load i32, i32* %6, align 4, !dbg !23
  store i32 %7, i32* %e, align 4, !dbg !23
  store i32 0, i32* @global_var, align 4, !dbg !26
  call void @__cxa_end_catch() #2, !dbg !28
  br label %try.cont, !dbg !28

try.cont:                                         ; preds = %catch, %invoke.cont
  store i32 1, i32* @global_var, align 4, !dbg !29
  ret i32 0, !dbg !30

eh.resume:                                        ; preds = %catch.dispatch
  %exn1 = load i8*, i8** %exn.slot, !dbg !23
  %sel2 = load i32, i32* %ehselector.slot, !dbg !23
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn1, 0, !dbg !23
  %lpad.val3 = insertvalue { i8*, i32 } %lpad.val, i32 %sel2, 1, !dbg !23
  resume { i8*, i32 } %lpad.val3, !dbg !23
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #1

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()

attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!31}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.3 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "<unknown>", directory: "")
!2 = !{}
!3 = !{!4, !10}
!4 = distinct !DISubprogram(name: "test", linkageName: "_Z4testi", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 4, file: !5, scope: !6, type: !7, variables: !2)
!5 = !DIFile(filename: "test.cpp", directory: "")
!6 = !DIFile(filename: "test.cpp", directory: "")
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !9}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = distinct !DISubprogram(name: "test2", linkageName: "_Z5test2v", line: 11, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 11, file: !5, scope: !6, type: !11, variables: !2)
!11 = !DISubroutineType(types: !12)
!12 = !{!9}
!13 = !DILocalVariable(name: "k", line: 4, arg: 1, scope: !4, file: !6, type: !9)
!14 = !DILocation(line: 4, scope: !4)
!15 = !DILocalVariable(name: "k2", line: 5, scope: !4, file: !6, type: !9)
!16 = !DILocation(line: 5, scope: !4)
!17 = !DILocation(line: 6, scope: !4)
!18 = !DILocation(line: 7, scope: !4)
!19 = !DILocation(line: 8, scope: !4)
!20 = !DILocation(line: 9, scope: !4)
!21 = !DILocation(line: 14, scope: !22)
!22 = distinct !DILexicalBlock(line: 13, column: 0, file: !5, scope: !10)
!23 = !DILocation(line: 15, scope: !22)
!24 = !DILocalVariable(name: "e", line: 16, scope: !10, file: !6, type: !9)
!25 = !DILocation(line: 16, scope: !10)
!26 = !DILocation(line: 17, scope: !27)
!27 = distinct !DILexicalBlock(line: 16, column: 0, file: !5, scope: !10)
!28 = !DILocation(line: 18, scope: !27)
!29 = !DILocation(line: 19, scope: !10)
!30 = !DILocation(line: 20, scope: !10)
!31 = !{i32 1, !"Debug Info Version", i32 3}
