; RUN: llc -filetype=obj -o - %s | llvm-dwarfdump - | FileCheck %s
; Generated from:
; void * __attribute__ (( regparm(2) )) f(void *, void *);
; void * __attribute__ (( regparm(0) )) g(void *, void *);
;  
; void *g(void *t, void *k) {
;   if (k == (void *)0)
;     return (void *)0;
;   return f(t, k);
; }

; CHECK: DW_TAG_formal_parameter
; CHECK-NEXT: DW_AT_location	(DW_OP_fbreg +4)
; CHECK-NEXT: DW_AT_name	("t")
; CHECK: DW_TAG_formal_parameter
; CHECK-NEXT: DW_AT_location	(DW_OP_fbreg +8)
; CHECK-NEXT: DW_AT_name	("k")

source_filename = "t.c"
target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386--linux-gnu"

; Function Attrs: nounwind
define i8* @g(i8* %t, i8* %k) local_unnamed_addr #0 !dbg !10 {
entry:
  call void @llvm.dbg.value(metadata i8* %t, metadata !14, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.value(metadata i8* %k, metadata !15, metadata !DIExpression()), !dbg !17
  %cmp = icmp eq i8* %k, null, !dbg !18
  br i1 %cmp, label %return, label %if.end, !dbg !20

if.end:                                           ; preds = %entry
  %call = tail call i8* @f(i8* inreg %t, i8* inreg nonnull %k) #3, !dbg !21
  br label %return, !dbg !22

return:                                           ; preds = %entry, %if.end
  %retval.0 = phi i8* [ %call, %if.end ], [ null, %entry ]
  ret i8* %retval.0, !dbg !23
}

declare i8* @f(i8* inreg, i8* inreg) local_unnamed_addr

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind  }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0  (llvm/trunk 319230)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 32)
!5 = !{i32 1, !"NumRegisterParameters", i32 0}
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{!"clang version 6.0.0  (llvm/trunk 319230)"}
!10 = distinct !DISubprogram(name: "g", scope: !1, file: !1, line: 4, type: !11, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !13)
!11 = !DISubroutineType(types: !12)
!12 = !{!4, !4, !4}
!13 = !{!14, !15}
!14 = !DILocalVariable(name: "t", arg: 1, scope: !10, file: !1, line: 4, type: !4)
!15 = !DILocalVariable(name: "k", arg: 2, scope: !10, file: !1, line: 4, type: !4)
!16 = !DILocation(line: 4, column: 15, scope: !10)
!17 = !DILocation(line: 4, column: 24, scope: !10)
!18 = !DILocation(line: 5, column: 9, scope: !19)
!19 = distinct !DILexicalBlock(scope: !10, file: !1, line: 5, column: 7)
!20 = !DILocation(line: 5, column: 7, scope: !10)
!21 = !DILocation(line: 7, column: 10, scope: !10)
!22 = !DILocation(line: 7, column: 3, scope: !10)
!23 = !DILocation(line: 8, column: 1, scope: !10)
