; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; bool f();
; inline __attribute__((always_inline)) int f1() {
;   if (bool b = f())
;     return 1;
;   return 2;
; }
;
; inline __attribute__((always_inline)) int f2() {
; # 2 "y.cc"
;   if (bool b = f())
;     return 3;
;   return 4;
; }
;
; int main() {
;   f1();
;   f2();
; }

; Ensure that lexical_blocks within inlined_subroutines are preserved/emitted.
; CHECK: DW_TAG_inlined_subroutine
; CHECK-NOT: DW_TAG
; CHECK-NOT: NULL
; CHECK: DW_TAG_lexical_block
; CHECK-NOT: DW_TAG
; CHECK-NOT: NULL
; CHECK: DW_TAG_variable
; Ensure that file changes don't interfere with creating inlined subroutines.
; (see the line directive inside 'f2' in thesource)
; CHECK: DW_TAG_inlined_subroutine
; CHECK:   DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_abstract_origin

; Function Attrs: uwtable
define i32 @main() #0 !dbg !4 {
entry:
  %retval.i2 = alloca i32, align 4
  %b.i3 = alloca i8, align 1
  %retval.i = alloca i32, align 4
  %b.i = alloca i8, align 1
  call void @llvm.dbg.declare(metadata i8* %b.i, metadata !16, metadata !DIExpression()), !dbg !19
  %call.i = call zeroext i1 @_Z1fv(), !dbg !19
  %frombool.i = zext i1 %call.i to i8, !dbg !19
  store i8 %frombool.i, i8* %b.i, align 1, !dbg !19
  %0 = load i8, i8* %b.i, align 1, !dbg !19
  %tobool.i = trunc i8 %0 to i1, !dbg !19
  br i1 %tobool.i, label %if.then.i, label %if.end.i, !dbg !19

if.then.i:                                        ; preds = %entry
  store i32 1, i32* %retval.i, !dbg !21
  br label %_Z2f1v.exit, !dbg !21

if.end.i:                                         ; preds = %entry
  store i32 2, i32* %retval.i, !dbg !22
  br label %_Z2f1v.exit, !dbg !22

_Z2f1v.exit:                                      ; preds = %if.then.i, %if.end.i
  %1 = load i32, i32* %retval.i, !dbg !23
  call void @llvm.dbg.declare(metadata i8* %b.i3, metadata !24, metadata !DIExpression()), !dbg !27
  %call.i4 = call zeroext i1 @_Z1fv(), !dbg !27
  %frombool.i5 = zext i1 %call.i4 to i8, !dbg !27
  store i8 %frombool.i5, i8* %b.i3, align 1, !dbg !27
  %2 = load i8, i8* %b.i3, align 1, !dbg !27
  %tobool.i6 = trunc i8 %2 to i1, !dbg !27
  br i1 %tobool.i6, label %if.then.i7, label %if.end.i8, !dbg !27

if.then.i7:                                       ; preds = %_Z2f1v.exit
  store i32 3, i32* %retval.i2, !dbg !29
  br label %_Z2f2v.exit, !dbg !29

if.end.i8:                                        ; preds = %_Z2f1v.exit
  store i32 4, i32* %retval.i2, !dbg !30
  br label %_Z2f2v.exit, !dbg !30

_Z2f2v.exit:                                      ; preds = %if.then.i7, %if.end.i8
  %3 = load i32, i32* %retval.i2, !dbg !31
  ret i32 0, !dbg !32
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare zeroext i1 @_Z1fv() #2

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14}
!llvm.ident = !{!15}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: 1, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "inline-scopes.cpp", directory: "/tmp/dbginfo")
!2 = !{}
!3 = !{!4, !10, !12}
!4 = distinct !DISubprogram(name: "main", line: 7, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 7, file: !5, scope: !6, type: !7, variables: !2)
!5 = !DIFile(filename: "y.cc", directory: "/tmp/dbginfo")
!6 = !DIFile(filename: "y.cc", directory: "/tmp/dbginfo")
!7 = !DISubroutineType(types: !8)
!8 = !{!9}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = distinct !DISubprogram(name: "f2", linkageName: "_Z2f2v", line: 8, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 8, file: !1, scope: !11, type: !7, variables: !2)
!11 = !DIFile(filename: "inline-scopes.cpp", directory: "/tmp/dbginfo")
!12 = distinct !DISubprogram(name: "f1", linkageName: "_Z2f1v", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 2, file: !1, scope: !11, type: !7, variables: !2)
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !{i32 1, !"Debug Info Version", i32 3}
!15 = !{!"clang version 3.5.0 "}
!16 = !DILocalVariable(name: "b", line: 3, scope: !17, file: !11, type: !18)
!17 = distinct !DILexicalBlock(line: 3, column: 0, file: !1, scope: !12)
!18 = !DIBasicType(tag: DW_TAG_base_type, name: "bool", size: 8, align: 8, encoding: DW_ATE_boolean)
!19 = !DILocation(line: 3, scope: !17, inlinedAt: !20)
!20 = !DILocation(line: 8, scope: !4)
!21 = !DILocation(line: 4, scope: !17, inlinedAt: !20)
!22 = !DILocation(line: 5, scope: !12, inlinedAt: !20)
!23 = !DILocation(line: 6, scope: !12, inlinedAt: !20)
!24 = !DILocalVariable(name: "b", line: 2, scope: !25, file: !6, type: !18)
!25 = distinct !DILexicalBlock(line: 2, column: 0, file: !5, scope: !26)
!26 = !DILexicalBlockFile(discriminator: 0, file: !5, scope: !10)
!27 = !DILocation(line: 2, scope: !25, inlinedAt: !28)
!28 = !DILocation(line: 9, scope: !4)
!29 = !DILocation(line: 3, scope: !25, inlinedAt: !28)
!30 = !DILocation(line: 4, scope: !26, inlinedAt: !28)
!31 = !DILocation(line: 5, scope: !26, inlinedAt: !28)
!32 = !DILocation(line: 10, scope: !4)
