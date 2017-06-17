; RUN: %llc_dwarf -filetype=obj < %s | llvm-dwarfdump - | FileCheck %s
; Generated at -O2 from:
; bool c();
; void f();
; bool start() {
;   bool result = c();
;   if (!c()) {
;     result = false;
;     goto exit;
;   }
;   f();
;   result = true;
; exit:
;   return result;
; }
;
; The constant should NOT be available for the entire function.
; CHECK-NOT: DW_AT_const_value
; CHECK: .debug_loc
; CHECK: Location description: 10 01 9f
;                              constu 0x00000001, stack-value
source_filename = "test.ii"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

; Function Attrs: noimplicitfloat noredzone nounwind optsize
define zeroext i1 @_Z5startv() local_unnamed_addr #0 !dbg !7 {
entry:
  %call = tail call zeroext i1 @_Z1cv() #3, !dbg !13
  %call1 = tail call zeroext i1 @_Z1cv() #3, !dbg !14
  br i1 %call1, label %if.end, label %exit, !dbg !16

if.end:                                           ; preds = %entry
  tail call void @_Z1fv() #3, !dbg !17
  tail call void @llvm.dbg.value(metadata i8 1, i64 0, metadata !12, metadata !18), !dbg !19
  br label %exit, !dbg !20

exit:                                             ; preds = %entry, %if.end
  %result.0 = phi i1 [ true, %if.end ], [ false, %entry ]
  ret i1 %result.0, !dbg !21
}

; Function Attrs: noimplicitfloat noredzone optsize
declare zeroext i1 @_Z1cv() local_unnamed_addr #1

; Function Attrs: noimplicitfloat noredzone optsize
declare void @_Z1fv() local_unnamed_addr #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { noimplicitfloat noredzone nounwind optsize }
attributes #1 = { noimplicitfloat noredzone optsize }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { nobuiltin noimplicitfloat noredzone nounwind optsize }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 5.0.0 (trunk 303873) (llvm/trunk 303897)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.ii", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 5.0.0 (trunk 303873) (llvm/trunk 303897)"}
!7 = distinct !DISubprogram(name: "start", linkageName: "_Z5startv", scope: !1, file: !1, line: 3, type: !8, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!11 = !{!12}
!12 = !DILocalVariable(name: "result", scope: !7, file: !1, line: 4, type: !10)
!13 = !DILocation(line: 4, column: 17, scope: !7)
!14 = !DILocation(line: 5, column: 8, scope: !15)
!15 = distinct !DILexicalBlock(scope: !7, file: !1, line: 5, column: 7)
!16 = !DILocation(line: 5, column: 7, scope: !7)
!17 = !DILocation(line: 9, column: 3, scope: !7)
!18 = !DIExpression()
!19 = !DILocation(line: 4, column: 8, scope: !7)
!20 = !DILocation(line: 10, column: 3, scope: !7)
!21 = !DILocation(line: 12, column: 3, scope: !7)
