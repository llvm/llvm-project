; RUN: llc -mtriple=x86_64-unknown-linux-gnu -stop-after=finalize-isel -o - %s \
; RUN:     -experimental-debug-variable-locations=true \
; RUN:   | FileCheck %s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -stop-after=finalize-isel -o - %s \
; RUN:     -experimental-debug-variable-locations=false \
; RUN:   | FileCheck %s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o - %s \
; RUN:   | llvm-dwarfdump --name=lazy -p - \
; RUN:   | FileCheck %s --check-prefix=DWARF

;; A C bool parameter is passed as i1 zeroext but stored as i8, so its
;; #dbg_value refers to the zext of the argument rather than to the argument
;; itself. Once lowered, the zext of a zeroext argument is no instruction at all
;; but the argument register, as is a trunc of an argument. If such a value is
;; the first location of a parameter, with an empty expression, describe it by
;; the argument register at the function entry, as the argument itself would be
;; (EmitFuncArgumentDbgValue).
;;
;; Generated with
;;   clang -O2 -g -S -emit-llvm -Wno-deprecated-non-prototype \
;;     -fdebug-compilation-dir=. t.c
;; and reduced by hand, from t.c:
;;
;;   void ext(void);
;;   void zext_bool(char *p, _Bool lazy) { ext(); *p = lazy; }
;;   void zext_bitint(char *p, unsigned _BitInt(4) ub) { ext(); *p = ub; }
;;   void sext_bitint(char *p, _BitInt(4) sb) { ext(); *p = sb; }
;;   void trunc_knr(p, s) short *p; short s; { ext(); *p = s; }
;;   void two_bools(char *p, _Bool ta, _Bool tb) { ext(); p[0] = ta; p[1] = tb; }
;;   void local_var(char *p, _Bool lv) { _Bool local = lv; ext(); *p = local; }
;;
;;   struct pair { int lo, hi; };
;;   void fragment(int *p, struct pair frag) { ext(); *p = frag.lo; }
;;
;;   void repeated_value(char *p, _Bool rep) {
;;     _Bool saved = rep;
;;     ext();
;;     rep = *p;
;;     ext();
;;     rep = saved;
;;     *p = rep;
;;   }

; CHECK-DAG: ![[LAZY:[0-9]+]] = !DILocalVariable(name: "lazy", arg: 2
; CHECK-DAG: ![[UB:[0-9]+]] = !DILocalVariable(name: "ub", arg: 2
; CHECK-DAG: ![[SB:[0-9]+]] = !DILocalVariable(name: "sb", arg: 2
; CHECK-DAG: ![[S:[0-9]+]] = !DILocalVariable(name: "s", arg: 2
; CHECK-DAG: ![[TA:[0-9]+]] = !DILocalVariable(name: "ta", arg: 2
; CHECK-DAG: ![[TB:[0-9]+]] = !DILocalVariable(name: "tb", arg: 3
; CHECK-DAG: ![[LV:[0-9]+]] = !DILocalVariable(name: "lv", arg: 2
; CHECK-DAG: ![[LOCAL:[0-9]+]] = !DILocalVariable(name: "local"
; CHECK-DAG: ![[FRAG:[0-9]+]] = !DILocalVariable(name: "frag", arg: 2
; CHECK-DAG: ![[REP:[0-9]+]] = !DILocalVariable(name: "rep", arg: 2
; CHECK-DAG: ![[TE:[0-9]+]] = !DILocalVariable(name: "te", arg: 2
; CHECK-DAG: ![[FO:[0-9]+]] = !DILocalVariable(name: "fo", arg: 2
; CHECK-DAG: ![[FE:[0-9]+]] = !DILocalVariable(name: "fe", arg: 2
; CHECK-DAG: ![[FV:[0-9]+]] = !DILocalVariable(name: "fv", arg: 2
; CHECK-DAG: ![[IC:[0-9]+]] = !DILocalVariable(name: "ic", arg: 2

;; The zext of the zeroext bool argument is the argument register: describe
;; 'lazy' by it at the function entry.
; CHECK-LABEL: name: zext_bool
; CHECK:         DBG_VALUE $esi, $noreg, ![[LAZY]], !DIExpression()
; CHECK-NEXT:    COPY $esi

; DWARF:      DW_TAG_subprogram
; DWARF-NEXT:   DW_AT_low_pc (0x[[ZB:[0-9a-f]+]])
; DWARF:      DW_AT_name ("zext_bool")
; DWARF:      DW_AT_location
; DWARF-NEXT:   [0x[[ZB]], {{.*}}): DW_OP_reg4 RSI
; DWARF:      DW_AT_name ("lazy")
define void @zext_bool(ptr %0, i1 zeroext %1) !dbg !7 {
  %3 = zext i1 %1 to i8, !dbg !9
    #dbg_value(i8 %3, !8, !DIExpression(), !9)
  call void @ext(), !dbg !9
  store i8 %3, ptr %0, !dbg !9
  ret void, !dbg !9
}

declare void @ext()

;; The zext of the zeroext _BitInt(4) argument is the argument register too.
; CHECK-LABEL: name: zext_bitint
; CHECK:         DBG_VALUE $esi, $noreg, ![[UB]], !DIExpression()
; CHECK-NEXT:    COPY $esi
define void @zext_bitint(ptr %0, i4 zeroext %1) !dbg !10 {
  %3 = zext i4 %1 to i8, !dbg !13
    #dbg_value(i8 %3, !12, !DIExpression(), !13)
  call void @ext(), !dbg !13
  store i8 %3, ptr %0, !dbg !13
  ret void, !dbg !13
}

;; The sext of the signext _BitInt(4) argument is the argument register too.
; CHECK-LABEL: name: sext_bitint
; CHECK:         DBG_VALUE $esi, $noreg, ![[SB]], !DIExpression()
; CHECK-NEXT:    COPY $esi
define void @sext_bitint(ptr %0, i4 signext %1) !dbg !14 {
  %3 = sext i4 %1 to i8, !dbg !17
    #dbg_value(i8 %3, !16, !DIExpression(), !17)
  call void @ext(), !dbg !17
  store i8 %3, ptr %0, !dbg !17
  ret void, !dbg !17
}

;; A K&R short parameter is passed as int: its trunc is the low part of the
;; argument register.
; CHECK-LABEL: name: trunc_knr
; CHECK:         DBG_VALUE $esi, $noreg, ![[S]], !DIExpression()
; CHECK-NEXT:    COPY $esi
define void @trunc_knr(ptr %0, i32 %1) !dbg !18 {
  %3 = trunc i32 %1 to i16, !dbg !21
    #dbg_value(i16 %3, !20, !DIExpression(), !21)
  call void @ext(), !dbg !21
  store i16 %3, ptr %0, !dbg !21
  ret void, !dbg !21
}

;; Each parameter is described by its own argument register.
; CHECK-LABEL: name: two_bools
; CHECK:         DBG_VALUE $esi, $noreg, ![[TA]], !DIExpression()
; CHECK-NEXT:    DBG_VALUE $edx, $noreg, ![[TB]], !DIExpression()
; CHECK-NEXT:    COPY $edx
define void @two_bools(ptr %0, i1 zeroext %1, i1 zeroext %2) !dbg !22 {
  %4 = zext i1 %1 to i8, !dbg !25
    #dbg_value(i8 %4, !23, !DIExpression(), !25)
  %5 = zext i1 %2 to i8, !dbg !25
    #dbg_value(i8 %5, !24, !DIExpression(), !25)
  call void @ext(), !dbg !25
  store i8 %4, ptr %0, !dbg !25
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 1, !dbg !25
  store i8 %5, ptr %6, !dbg !25
  ret void, !dbg !25
}

;; Not a parameter: 'local' stays where it is, although it has the value of
;; 'lv', which is described at the function entry.
; CHECK-LABEL: name: local_var
; CHECK:         DBG_VALUE $esi, $noreg, ![[LV]], !DIExpression()
; CHECK-NOT:     ![[LOCAL]]
; CHECK:         COPY $esi
; CHECK:         ![[LOCAL]]
define void @local_var(ptr %0, i1 zeroext %1) !dbg !26 {
  %3 = zext i1 %1 to i8, !dbg !29
    #dbg_value(i8 %3, !27, !DIExpression(), !29)
    #dbg_value(i8 %3, !28, !DIExpression(), !29)
  call void @ext(), !dbg !29
  store i8 %3, ptr %0, !dbg !29
  ret void, !dbg !29
}

;; A fragment is left alone: the trunc for 'frag.lo' stays where it is.
; CHECK-LABEL: name: fragment
; CHECK-NOT:     ![[FRAG]], !DIExpression({{(DW_OP_LLVM_arg, 0, )?}}DW_OP_LLVM_fragment, 0, 32)
; CHECK:         COPY $rsi
; CHECK:         ![[FRAG]], !DIExpression({{(DW_OP_LLVM_arg, 0, )?}}DW_OP_LLVM_fragment, 0, 32)
define void @fragment(ptr %0, i64 %1) !dbg !30 {
  %3 = trunc i64 %1 to i32, !dbg !37
    #dbg_value(i32 %3, !36, !DIExpression(DW_OP_LLVM_fragment, 0, 32), !37)
  call void @ext(), !dbg !37
  store i32 %3, ptr %0, !dbg !37
  ret void, !dbg !37
}

;; Only the first location of the parameter is described at the function
;; entry: the later location of the same value stays where it is, as the
;; argument is already described.
; CHECK-LABEL: name: repeated_value
; CHECK:         DBG_VALUE $esi, $noreg, ![[REP]], !DIExpression()
; CHECK-NEXT:    COPY $esi
; CHECK:         CALL64pcrel32
; CHECK:         DBG_VALUE $noreg, $noreg, ![[REP]], !DIExpression()
; CHECK:         CALL64pcrel32
; CHECK-NEXT:    ADJCALLSTACKUP64
; CHECK-NEXT:    DBG_{{.*}}![[REP]], !DIExpression(
define void @repeated_value(ptr %0, i1 zeroext %1) !dbg !38 {
  %3 = zext i1 %1 to i8, !dbg !40
    #dbg_value(i8 %3, !39, !DIExpression(), !40)
  call void @ext(), !dbg !40
    #dbg_value(i8 poison, !39, !DIExpression(), !40)
  call void @ext(), !dbg !40
    #dbg_value(i8 %3, !39, !DIExpression(), !40)
  store i8 %3, ptr %0, !dbg !40
  ret void, !dbg !40
}

;; The cases below are hand-written, not from t.c.

;; The location of 'te' has an expression: the trunc stays where it is, as the
;; expression would use the whole register, which is wider than the value.
; CHECK-LABEL: name: trunc_expr
; CHECK-NOT:     ![[TE]], !DIExpression({{(DW_OP_LLVM_arg, 0, )?}}DW_OP_constu, 1, DW_OP_shr, DW_OP_stack_value)
; CHECK:         COPY $rsi
; CHECK:         ![[TE]], !DIExpression({{(DW_OP_LLVM_arg, 0, )?}}DW_OP_constu, 1, DW_OP_shr, DW_OP_stack_value)
define void @trunc_expr(ptr %0, i64 %1) !dbg !41 {
  %3 = trunc i64 %1 to i32, !dbg !43
    #dbg_value(i32 %3, !42, !DIExpression(DW_OP_constu, 1, DW_OP_shr, DW_OP_stack_value), !43)
  call void @ext(), !dbg !43
  store i32 %3, ptr %0, !dbg !43
  ret void, !dbg !43
}

;; The first location of 'fo' is a constant: the later location of the zext
;; stays where it is, instead of moving before the constant.
; CHECK-LABEL: name: first_other
; CHECK-NOT:     DBG_VALUE $esi, $noreg, ![[FO]]
; CHECK:         DBG_VALUE 0, $noreg, ![[FO]], !DIExpression()
; CHECK:         CALL64pcrel32
; CHECK:         DBG_{{.*}}![[FO]], !DIExpression(
define void @first_other(ptr %0, i1 zeroext %1) !dbg !44 {
    #dbg_value(i8 0, !45, !DIExpression(), !46)
  %3 = zext i1 %1 to i8, !dbg !46
  call void @ext(), !dbg !46
    #dbg_value(i8 %3, !45, !DIExpression(), !46)
  store i8 %3, ptr %0, !dbg !46
  ret void, !dbg !46
}

;; The first location of 'fe' is the zext with an expression: the later
;; location of the zext, with none, stays where it is too.
; CHECK-LABEL: name: first_expr
; CHECK-NOT:     DBG_VALUE $esi, $noreg, ![[FE]]
; CHECK:         ![[FE]], !DIExpression({{(DW_OP_LLVM_arg, 0, )?}}DW_OP_plus_uconst, 1, DW_OP_stack_value)
; CHECK:         CALL64pcrel32
; CHECK:         DBG_{{.*}}![[FE]], !DIExpression(
define void @first_expr(ptr %0, i1 zeroext %1) !dbg !47 {
  %3 = zext i1 %1 to i8, !dbg !49
    #dbg_value(i8 %3, !48, !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value), !49)
  call void @ext(), !dbg !49
    #dbg_value(i8 %3, !48, !DIExpression(), !49)
  store i8 %3, ptr %0, !dbg !49
  ret void, !dbg !49
}

;; The first location of 'fv' is variadic: the later location of the zext
;; stays where it is too, although the zext is the first operand.
; CHECK-LABEL: name: first_variadic
; CHECK-NOT:     DBG_VALUE $esi, $noreg, ![[FV]]
; CHECK:         ![[FV]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value)
; CHECK:         CALL64pcrel32
; CHECK:         DBG_{{.*}}![[FV]], !DIExpression(
define void @first_variadic(ptr %0, i1 zeroext %1) !dbg !50 {
  %3 = zext i1 %1 to i8, !dbg !52
    #dbg_value(!DIArgList(i8 %3, i8 1), !51, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value), !52)
  call void @ext(), !dbg !52
    #dbg_value(i8 %3, !51, !DIExpression(), !52)
  store i8 %3, ptr %0, !dbg !52
  ret void, !dbg !52
}

;; An inlined copy of 'ic' is not a location of the parameter: the zext is
;; still its first location, and is described at the function entry.
; CHECK-LABEL: name: inlined_copy
; CHECK:         DBG_VALUE $esi, $noreg, ![[IC]], !DIExpression()
; CHECK-NEXT:    COPY $esi
define void @inlined_copy(ptr %0, i1 zeroext %1) !dbg !53 {
    #dbg_value(i8 0, !54, !DIExpression(), !56)
  %3 = zext i1 %1 to i8, !dbg !55
    #dbg_value(i8 %3, !54, !DIExpression(), !55)
  call void @ext(), !dbg !55
  store i8 %3, ptr %0, !dbg !55
  ret void, !dbg !55
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: ".")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !5)
!5 = !{null}
!6 = !DIBasicType(name: "_Bool", size: 8, encoding: DW_ATE_boolean)
!7 = distinct !DISubprogram(name: "zext_bool", scope: !1, file: !1, line: 3, type: !4, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!8 = !DILocalVariable(name: "lazy", arg: 2, scope: !7, file: !1, line: 3, type: !6)
!9 = !DILocation(line: 0, scope: !7)
!10 = distinct !DISubprogram(name: "zext_bitint", scope: !1, file: !1, line: 5, type: !4, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DIBasicType(name: "unsigned _BitInt(4)", size: 8, dataSize: 4, encoding: DW_ATE_unsigned)
!12 = !DILocalVariable(name: "ub", arg: 2, scope: !10, file: !1, line: 5, type: !11)
!13 = !DILocation(line: 0, scope: !10)
!14 = distinct !DISubprogram(name: "sext_bitint", scope: !1, file: !1, line: 7, type: !4, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!15 = !DIBasicType(name: "_BitInt(4)", size: 8, dataSize: 4, encoding: DW_ATE_signed)
!16 = !DILocalVariable(name: "sb", arg: 2, scope: !14, file: !1, line: 7, type: !15)
!17 = !DILocation(line: 0, scope: !14)
!18 = distinct !DISubprogram(name: "trunc_knr", scope: !1, file: !1, line: 9, type: !4, scopeLine: 9, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!19 = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
!20 = !DILocalVariable(name: "s", arg: 2, scope: !18, file: !1, line: 9, type: !19)
!21 = !DILocation(line: 0, scope: !18)
!22 = distinct !DISubprogram(name: "two_bools", scope: !1, file: !1, line: 11, type: !4, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!23 = !DILocalVariable(name: "ta", arg: 2, scope: !22, file: !1, line: 11, type: !6)
!24 = !DILocalVariable(name: "tb", arg: 3, scope: !22, file: !1, line: 11, type: !6)
!25 = !DILocation(line: 0, scope: !22)
!26 = distinct !DISubprogram(name: "local_var", scope: !1, file: !1, line: 13, type: !4, scopeLine: 13, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!27 = !DILocalVariable(name: "lv", arg: 2, scope: !26, file: !1, line: 13, type: !6)
!28 = !DILocalVariable(name: "local", scope: !26, file: !1, line: 13, type: !6)
!29 = !DILocation(line: 0, scope: !26)
!30 = distinct !DISubprogram(name: "fragment", scope: !1, file: !1, line: 16, type: !4, scopeLine: 16, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!31 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "pair", file: !1, line: 15, size: 64, elements: !32)
!32 = !{!33, !34}
!33 = !DIDerivedType(tag: DW_TAG_member, name: "lo", scope: !31, file: !1, line: 15, baseType: !35, size: 32)
!34 = !DIDerivedType(tag: DW_TAG_member, name: "hi", scope: !31, file: !1, line: 15, baseType: !35, size: 32, offset: 32)
!35 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!36 = !DILocalVariable(name: "frag", arg: 2, scope: !30, file: !1, line: 16, type: !31)
!37 = !DILocation(line: 0, scope: !30)
!38 = distinct !DISubprogram(name: "repeated_value", scope: !1, file: !1, line: 18, type: !4, scopeLine: 18, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!39 = !DILocalVariable(name: "rep", arg: 2, scope: !38, file: !1, line: 18, type: !6)
!40 = !DILocation(line: 0, scope: !38)
!41 = distinct !DISubprogram(name: "trunc_expr", scope: !1, file: !1, line: 27, type: !4, scopeLine: 27, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!42 = !DILocalVariable(name: "te", arg: 2, scope: !41, file: !1, line: 27, type: !35)
!43 = !DILocation(line: 0, scope: !41)
!44 = distinct !DISubprogram(name: "first_other", scope: !1, file: !1, line: 28, type: !4, scopeLine: 28, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!45 = !DILocalVariable(name: "fo", arg: 2, scope: !44, file: !1, line: 28, type: !6)
!46 = !DILocation(line: 0, scope: !44)
!47 = distinct !DISubprogram(name: "first_expr", scope: !1, file: !1, line: 29, type: !4, scopeLine: 29, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!48 = !DILocalVariable(name: "fe", arg: 2, scope: !47, file: !1, line: 29, type: !6)
!49 = !DILocation(line: 0, scope: !47)
!50 = distinct !DISubprogram(name: "first_variadic", scope: !1, file: !1, line: 30, type: !4, scopeLine: 30, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!51 = !DILocalVariable(name: "fv", arg: 2, scope: !50, file: !1, line: 30, type: !6)
!52 = !DILocation(line: 0, scope: !50)
!53 = distinct !DISubprogram(name: "inlined_copy", scope: !1, file: !1, line: 31, type: !4, scopeLine: 31, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!54 = !DILocalVariable(name: "ic", arg: 2, scope: !53, file: !1, line: 31, type: !6)
!55 = !DILocation(line: 0, scope: !53)
!56 = !DILocation(line: 0, scope: !53, inlinedAt: !55)
