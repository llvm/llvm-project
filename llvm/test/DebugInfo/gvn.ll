; RUN: opt < %s -O2 -gvn -S | FileCheck %s
;
; Produced at -O2 from:
; struct context {
;   int cur_pid
; };
; int a, b, c, f, d;
; int pid_for_task(int);
; sample(struct context *p1)
; {
;   if (c)
;     b = a;
;   if (a && p1->cur_pid)
;     sample_internal();
; }
; callback() {
;   f = pid_for_task(d);
;   sample(&f);
; }

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios"

%struct.context = type { i32 }

@c = common global i32 0, align 4
@a = common global i32 0, align 4
@b = common global i32 0, align 4
@d = common global i32 0, align 4
@f = common global i32 0, align 4

; Function Attrs: nounwind
declare i32 @sample_internal(...)

; Function Attrs: nounwind
define i32 @callback() #0 {
entry:
  %0 = load i32, i32* @d, align 4, !dbg !37

  ; Verify that the call still has a debug location after GVN.
  ; CHECK: %call = tail call i32 @pid_for_task(i32 %0) #{{[0-9]}}, !dbg
  %call = tail call i32 @pid_for_task(i32 %0) #3, !dbg !37

  store i32 %call, i32* @f, align 4, !dbg !37
  tail call void @llvm.dbg.value(metadata %struct.context* bitcast (i32* @f to %struct.context*), i64 0, metadata !25, metadata !26) #3, !dbg !38
  %1 = load i32, i32* @c, align 4, !dbg !40
  %tobool.i = icmp eq i32 %1, 0, !dbg !40
  %.pr.i = load i32, i32* @a, align 4, !dbg !41
  br i1 %tobool.i, label %if.end.i, label %if.then.i, !dbg !42

if.then.i:                                        ; preds = %entry
  store i32 %.pr.i, i32* @b, align 4, !dbg !43
  br label %if.end.i, !dbg !43

if.end.i:                                         ; preds = %if.then.i, %entry
  %tobool1.i = icmp eq i32 %.pr.i, 0, !dbg !41

  ; This instruction has no debug location -- in this
  ; particular case it was removed by a bug in SimplifyCFG.
  %2 = load i32, i32* @f, align 4

  ; GVN is supposed to replace the load of @f with a direct reference to %call.
  ; CHECK: %tobool2.i = icmp eq i32 %call, 0, !dbg
  %tobool2.i = icmp eq i32 %2, 0, !dbg !41

  %or.cond = or i1 %tobool1.i, %tobool2.i, !dbg !41
  br i1 %or.cond, label %sample.exit, label %if.then.3.i, !dbg !41

if.then.3.i:                                      ; preds = %if.end.i
  %call.i = tail call i32 bitcast (i32 (...)* @sample_internal to i32 ()*)() #3, !dbg !44
  br label %sample.exit, !dbg !44

sample.exit:                                      ; preds = %if.end.i, %if.then.3.i
  ret i32 undef, !dbg !45
}

declare i32 @pid_for_task(i32) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { nounwind }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22, !23}
!llvm.ident = !{!24}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0 (trunk 244473) (llvm/trunk 244644)", isOptimized: false, runtimeVersion: 0, emissionKind: 1, enums: !2, subprograms: !3, globals: !16)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !{!4, !13}
!4 = !DISubprogram(name: "sample", scope: !5, file: !5, line: 6, type: !6, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: false, variables: !2)
!5 = !DIFile(filename: "test.i", directory: "/")
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9}
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64, align: 64)
!10 = !DICompositeType(tag: DW_TAG_structure_type, name: "context", file: !5, line: 1, size: 32, align: 32, elements: !11)
!11 = !{!12}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "cur_pid", scope: !10, file: !5, line: 2, baseType: !8, size: 32, align: 32)
!13 = !DISubprogram(name: "callback", scope: !5, file: !5, line: 13, type: !14, isLocal: false, isDefinition: true, scopeLine: 13, isOptimized: false, function: i32 ()* @callback, variables: !2)
!14 = !DISubroutineType(types: !15)
!15 = !{!8}
!16 = !{!17, !18, !19, !20, !21}
!17 = !DIGlobalVariable(name: "a", scope: !0, file: !5, line: 4, type: !8, isLocal: false, isDefinition: true, variable: i32* @a)
!18 = !DIGlobalVariable(name: "b", scope: !0, file: !5, line: 4, type: !8, isLocal: false, isDefinition: true, variable: i32* @b)
!19 = !DIGlobalVariable(name: "c", scope: !0, file: !5, line: 4, type: !8, isLocal: false, isDefinition: true, variable: i32* @c)
!20 = !DIGlobalVariable(name: "f", scope: !0, file: !5, line: 4, type: !8, isLocal: false, isDefinition: true, variable: i32* @f)
!21 = !DIGlobalVariable(name: "d", scope: !0, file: !5, line: 4, type: !8, isLocal: false, isDefinition: true, variable: i32* @d)
!22 = !{i32 2, !"Dwarf Version", i32 2}
!23 = !{i32 2, !"Debug Info Version", i32 3}
!24 = !{!"clang version 3.8.0 (trunk 244473) (llvm/trunk 244644)"}
!25 = !DILocalVariable(tag: DW_TAG_auto_variable, name: "p1", arg: 1, scope: !4, file: !5, line: 6, type: !9)
!26 = !DIExpression()
!27 = !DILocation(line: 6, scope: !4)
!28 = !DILocation(line: 8, scope: !29)
!29 = distinct !DILexicalBlock(scope: !4, file: !5, line: 8)
!30 = !DILocation(line: 10, scope: !31)
!31 = distinct !DILexicalBlock(scope: !4, file: !5, line: 10)
!32 = !DILocation(line: 8, scope: !4)
!33 = !DILocation(line: 9, scope: !29)
!34 = !DILocation(line: 10, scope: !4)
!35 = !DILocation(line: 11, scope: !31)
!36 = !DILocation(line: 12, scope: !4)
!37 = !DILocation(line: 14, scope: !13)
!38 = !DILocation(line: 6, scope: !4, inlinedAt: !39)
!39 = distinct !DILocation(line: 15, scope: !13)
!40 = !DILocation(line: 8, scope: !29, inlinedAt: !39)
!41 = !DILocation(line: 10, scope: !31, inlinedAt: !39)
!42 = !DILocation(line: 8, scope: !4, inlinedAt: !39)
!43 = !DILocation(line: 9, scope: !29, inlinedAt: !39)
!44 = !DILocation(line: 11, scope: !31, inlinedAt: !39)
!45 = !DILocation(line: 16, scope: !13)
