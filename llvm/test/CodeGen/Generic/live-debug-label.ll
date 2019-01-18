; RUN: llc < %s -stop-after=virtregrewriter -o - | FileCheck %s
;
; Generated with "clang++ -g -O1 -S -emit-llvm"
;
; inline bool bar(char c) {
;   return c >= '0' && c <= '9';
; }
;
; unsigned foo(const char* data,
;              int length,
;              int* parsing_result) {
;   unsigned value = 0;
;   int result = 1;
;   bool overflow = 0;
;
;   while (bar(*data)) {
;     if (value > 1) {
;       result = 2;
;       overflow = 1;
;     }
;
;     if (!overflow)
;       value = value + 1;
;   }
;
;   if (length == 0 || value) {
;     if (!overflow)
;       result = 0;
;   } else {
;     result = 1;
;   }
; bye:
;   *parsing_result = result;
;   return result == 0 ? value : 0;
; }
;
; CHECK: {{^body:}}
; CHECK: bye.thread21:
; CHECK: DBG_LABEL !14
; CHECK: if.then5:
; CHECK: DBG_LABEL !14
; CHECK-NOT: DBG_LABEL !14

$_Z3barc = comdat any

; Function Attrs: nounwind uwtable
define dso_local i32 @_Z3fooPKciPi(i8* nocapture readonly %data, i32 %length, i32* nocapture %parsing_result) local_unnamed_addr !dbg !4 {
entry:
  %0 = load i8, i8* %data, align 1
  %call23 = tail call zeroext i1 @_Z3barc(i8 signext %0), !dbg !15
  br i1 %call23, label %while.body, label %while.end

while.body:                                       ; preds = %entry, %while.body
  %overflow.026 = phi i8 [ %spec.select18, %while.body ], [ 0, %entry ]
  %result.025 = phi i32 [ %spec.select, %while.body ], [ 1, %entry ]
  %value.024 = phi i32 [ %value.1, %while.body ], [ 0, %entry ]
  %cmp = icmp ugt i32 %value.024, 1
  %spec.select = select i1 %cmp, i32 2, i32 %result.025
  %spec.select18 = select i1 %cmp, i8 1, i8 %overflow.026
  %1 = and i8 %spec.select18, 1
  %2 = xor i8 %1, 1
  %3 = zext i8 %2 to i32
  %value.1 = add i32 %value.024, %3
  %4 = load i8, i8* %data, align 1
  %call = tail call zeroext i1 @_Z3barc(i8 signext %4), !dbg !15
  br i1 %call, label %while.body, label %while.end.loopexit

while.end.loopexit:                               ; preds = %while.body
  %phitmp = and i8 %spec.select18, 1
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  %value.0.lcssa = phi i32 [ 0, %entry ], [ %value.1, %while.end.loopexit ]
  %result.0.lcssa = phi i32 [ 1, %entry ], [ %spec.select, %while.end.loopexit ]
  %overflow.0.lcssa = phi i8 [ 0, %entry ], [ %phitmp, %while.end.loopexit ]
  %cmp3 = icmp eq i32 %length, 0
  %tobool4 = icmp ne i32 %value.0.lcssa, 0
  %or.cond = or i1 %cmp3, %tobool4
  br i1 %or.cond, label %if.then5, label %bye.thread21

bye.thread21:                                     ; preds = %while.end
  call void @llvm.dbg.label(metadata !14), !dbg !16
  store i32 1, i32* %parsing_result, align 4
  br label %6

if.then5:                                         ; preds = %while.end
  %tobool6 = icmp eq i8 %overflow.0.lcssa, 0
  call void @llvm.dbg.label(metadata !14), !dbg !16
  call void @llvm.dbg.label(metadata !14), !dbg !16
  br i1 %tobool6, label %bye.thread, label %bye

bye.thread:                                       ; preds = %if.then5
  store i32 0, i32* %parsing_result, align 4
  br label %5

bye:                                              ; preds = %if.then5
  store i32 %result.0.lcssa, i32* %parsing_result, align 4
  %cmp10 = icmp eq i32 %result.0.lcssa, 0
  br i1 %cmp10, label %5, label %6

; <label>:5:                                      ; preds = %bye.thread, %bye
  br label %6

; <label>:6:                                      ; preds = %bye.thread21, %bye, %5
  %7 = phi i32 [ %value.0.lcssa, %5 ], [ 0, %bye ], [ 0, %bye.thread21 ]
  ret i32 %7
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local zeroext i1 @_Z3barc(i8 signext %c) local_unnamed_addr comdat {
entry:
  %c.off = add i8 %c, -48
  %0 = icmp ult i8 %c.off, 10
  ret i1 %0
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.label(metadata) #0

attributes #0 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "live-debug-label.cc", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooPKciPi", scope: !1, file: !1, line: 5, type: !5, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !13)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !8, !11, !12}
!7 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!9 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !10)
!10 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!13 = !{!14}
!14 = !DILabel(scope: !4, name: "bye", file: !1, line: 28)
!15 = !DILocation(line: 12, column: 10, scope: !4)
!16 = !DILocation(line: 28, column: 1, scope: !4)
