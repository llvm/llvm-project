; RUN: llc -mtriple=x86_64-unknown-linux-gnu -stop-after=finalize-isel -o - %s -experimental-debug-variable-locations=true | FileCheck %s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o - %s -experimental-debug-variable-locations=true | llvm-dwarfdump --name=lazy - | FileCheck %s --check-prefix=DWARF
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o - %s -experimental-debug-variable-locations=true | llvm-dwarfdump --name=two_bools -c - | FileCheck %s --check-prefix=DWARF-TWO

; A C bool parameter is passed as i1 zeroext but stored as i8, so its
; dbg.value refers to a zext of the argument rather than to the argument.
; Instruction selection turns the zext into a copy of the argument register,
; so the DBG_INSTR_REF refers to the DBG_PHI of that register. It is placed
; after the copies of the argument registers, which end up past the prologue.
; As the first location of the parameter is a value live into the function,
; hoist its DBG_INSTR_REF to the start of the function so that the parameter
; has a location at the function entry, as it would for an int parameter.
;
;   void ext(void);
;   void zext_bool(char *p, _Bool lazy) { ext(); *p = lazy; }

; CHECK: ![[LAZY:[0-9]+]] = !DILocalVariable(name: "lazy", arg: 2
; CHECK: ![[COMPLEX:[0-9]+]] = !DILocalVariable(name: "complex", arg: 2
; CHECK: ![[LATER:[0-9]+]] = !DILocalVariable(name: "later", arg: 1
; CHECK: ![[LOCAL:[0-9]+]] = !DILocalVariable(name: "local"
; CHECK: ![[OTHER:[0-9]+]] = !DILocalVariable(name: "a", arg: 1
; CHECK: ![[TA:[0-9]+]] = !DILocalVariable(name: "ta", arg: 2
; CHECK: ![[TB:[0-9]+]] = !DILocalVariable(name: "tb", arg: 3
; CHECK: ![[SA:[0-9]+]] = !DILocalVariable(name: "sa", arg: 2
; CHECK: ![[SB:[0-9]+]] = !DILocalVariable(name: "sb", arg: 3
; CHECK: ![[INL:[0-9]+]] = !DILocalVariable(name: "inl", arg: 1
; CHECK: ![[FRAG:[0-9]+]] = !DILocalVariable(name: "frag", arg: 2
; CHECK: ![[NC:[0-9]+]] = !DILocalVariable(name: "nc", arg: 2

; CHECK-LABEL: name: zext_bool
; CHECK:      debugValueSubstitutions:
; CHECK-NEXT:   - { srcinst: [[SUB:[0-9]+]], srcop: 0, dstinst: [[PHI:[0-9]+]], dstop: 0, subreg: 1 }
; CHECK:      bb.0.entry:
; CHECK-NEXT:   liveins:
; CHECK:        DBG_PHI $esi, [[PHI]]
; CHECK-NEXT:   DBG_INSTR_REF ![[LAZY]], !DIExpression(DW_OP_LLVM_arg, 0), dbg-instr-ref([[SUB]], 0)
; CHECK-NEXT:   COPY $esi

; DWARF:      DW_AT_location
; DWARF-NEXT: [0x0000000000000000, {{.*}}): DW_OP_reg4 RSI
; DWARF:      DW_AT_name ("lazy")

;; A complex expression may describe a later assignment to the parameter
;; rather than its incoming value: leave it in place.
; CHECK-LABEL: name: zext_bool_complex
; CHECK:      bb.0.entry:
; CHECK-NEXT:   liveins:
; CHECK:        DBG_PHI $esi
; CHECK-NEXT:   COPY $esi
; CHECK:        DBG_INSTR_REF ![[COMPLEX]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_plus_uconst, 1)
; CHECK-NEXT:   ADJCALLSTACKDOWN64

;; Only the first location of a parameter in the entry block is hoisted.
; CHECK-LABEL: name: later_location
; CHECK:      bb.0.entry:
; CHECK:        CALL64pcrel32
; CHECK:        DBG_INSTR_REF ![[LATER]], !DIExpression(DW_OP_LLVM_arg, 0)
; CHECK-NEXT:   MOV32mr

;; Only parameters are hoisted.
; CHECK-LABEL: name: local_var
; CHECK:      bb.0.entry:
; CHECK-NEXT:   liveins:
; CHECK:        DBG_PHI $esi
; CHECK-NEXT:   COPY $esi
; CHECK:        DBG_INSTR_REF ![[LOCAL]], !DIExpression(DW_OP_LLVM_arg, 0)
; CHECK-NEXT:   ADJCALLSTACKDOWN64

;; A parameter first located in the register of another parameter was likely
;; assigned from it, a = (int)b here, with the location of its incoming value
;; dropped: leave it in place, as EmitFuncArgumentDbgValue() does for a
;; dbg.value of an Argument.
; CHECK-LABEL: name: other_param
; CHECK:      bb.0.entry:
; CHECK-NEXT:   liveins:
; CHECK:        DBG_PHI $rsi
; CHECK-NOT:    DBG_INSTR_REF ![[OTHER]]
; CHECK:        COPY $rsi
; CHECK:        DBG_INSTR_REF ![[OTHER]], !DIExpression(DW_OP_LLVM_arg, 0)
; CHECK-NEXT:   ADJCALLSTACKDOWN64

;; Several parameters are hoisted, in their original order.
; CHECK-LABEL: name: two_bools
; CHECK:      bb.0.entry:
; CHECK-NEXT:   liveins:
; CHECK:        DBG_PHI $edx
; CHECK-NEXT:   DBG_PHI $esi
; CHECK-NEXT:   DBG_INSTR_REF ![[TA]], !DIExpression(DW_OP_LLVM_arg, 0)
; CHECK-NEXT:   DBG_INSTR_REF ![[TB]], !DIExpression(DW_OP_LLVM_arg, 0)
; CHECK-NEXT:   COPY $edx

; DWARF-TWO:      DW_AT_low_pc (0x[[ENTRY:[0-9a-f]+]])
; DWARF-TWO:      DW_AT_name ("two_bools")
; DWARF-TWO:      DW_AT_location
; DWARF-TWO-NEXT: [0x[[ENTRY]], {{.*}}): DW_OP_reg4 RSI
; DWARF-TWO:      DW_AT_name ("ta")
; DWARF-TWO:      DW_AT_location
; DWARF-TWO-NEXT: [0x[[ENTRY]], {{.*}}): DW_OP_reg1 RDX
; DWARF-TWO:      DW_AT_name ("tb")

;; If two parameters are first located in the same argument register, it is
;; unknown which one it belongs to: hoist neither.
; CHECK-LABEL: name: shared_reg
; CHECK:      bb.0.entry:
; CHECK-NEXT:   liveins:
; CHECK:        DBG_PHI $edx
; CHECK-NEXT:   COPY $edx
; CHECK:        DBG_INSTR_REF ![[SA]], !DIExpression(DW_OP_LLVM_arg, 0)
; CHECK-NEXT:   DBG_INSTR_REF ![[SB]], !DIExpression(DW_OP_LLVM_arg, 0)
; CHECK-NEXT:   ADJCALLSTACKDOWN64

;; A parameter of an inlined function is not a parameter of this function.
; CHECK-LABEL: name: inlined
; CHECK:      bb.0.entry:
; CHECK-NEXT:   liveins:
; CHECK:        DBG_PHI $esi
; CHECK-NEXT:   COPY $esi
; CHECK:        DBG_INSTR_REF ![[INL]], !DIExpression(DW_OP_LLVM_arg, 0)
; CHECK-NEXT:   ADJCALLSTACKDOWN64

;; A fragment is a plain location.
; CHECK-LABEL: name: fragment
; CHECK:      bb.0.entry:
; CHECK-NEXT:   liveins:
; CHECK:        DBG_PHI $rsi
; CHECK-NEXT:   DBG_INSTR_REF ![[FRAG]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_fragment, 0, 32)
; CHECK-NEXT:   COPY $rsi

;; A value computed from an argument is not live into the function.
; CHECK-LABEL: name: not_a_copy
; CHECK:      bb.0.entry:
; CHECK-NOT:    DBG_PHI
; CHECK:        INC32r
; CHECK-NEXT:   DBG_INSTR_REF ![[NC]], !DIExpression(DW_OP_LLVM_arg, 0)
; CHECK-NEXT:   ADJCALLSTACKDOWN64

declare void @ext()

define void @zext_bool(ptr %p, i1 zeroext %lazy) !dbg !7 {
entry:
  %frombool = zext i1 %lazy to i8
    #dbg_value(i8 %frombool, !13, !DIExpression(), !15)
  call void @ext(), !dbg !16
  store i8 %frombool, ptr %p, align 1, !dbg !16
  ret void, !dbg !16
}

define void @zext_bool_complex(ptr %p, i1 zeroext %lazy) !dbg !20 {
entry:
  %frombool = zext i1 %lazy to i8
    #dbg_value(i8 %frombool, !21, !DIExpression(DW_OP_plus_uconst, 1), !22)
  call void @ext(), !dbg !23
  store i8 %frombool, ptr %p, align 1, !dbg !23
  ret void, !dbg !23
}

define void @later_location(i32 %a, i64 %b, ptr %p) !dbg !30 {
entry:
    #dbg_value(i32 %a, !31, !DIExpression(), !32)
  %t = trunc i64 %b to i32
  call void @ext(), !dbg !33
    #dbg_value(i32 %t, !31, !DIExpression(), !32)
  store i32 %t, ptr %p, align 4, !dbg !33
  ret void, !dbg !33
}

define void @local_var(ptr %p, i1 zeroext %lazy) !dbg !40 {
entry:
  %frombool = zext i1 %lazy to i8
    #dbg_value(i8 %frombool, !41, !DIExpression(), !42)
  call void @ext(), !dbg !43
  store i8 %frombool, ptr %p, align 1, !dbg !43
  ret void, !dbg !43
}

define void @other_param(i32 %a, i64 %b, ptr %p) !dbg !50 {
entry:
    #dbg_value(i64 %b, !52, !DIExpression(), !53)
  %t = trunc i64 %b to i32
    #dbg_value(i32 %t, !51, !DIExpression(), !53)
  call void @ext(), !dbg !54
  store i32 %t, ptr %p, align 4, !dbg !54
  ret void, !dbg !54
}

define void @two_bools(ptr %p, i1 zeroext %a, i1 zeroext %b) !dbg !60 {
entry:
  %fa = zext i1 %a to i8
    #dbg_value(i8 %fa, !61, !DIExpression(), !63)
  %fb = zext i1 %b to i8
    #dbg_value(i8 %fb, !62, !DIExpression(), !63)
  call void @ext(), !dbg !64
  store i8 %fa, ptr %p, align 1, !dbg !64
  %q = getelementptr i8, ptr %p, i64 1
  store i8 %fb, ptr %q, align 1, !dbg !64
  ret void, !dbg !64
}

define void @shared_reg(ptr %p, i1 zeroext %a, i1 zeroext %b) !dbg !70 {
entry:
  %fb = zext i1 %b to i8
    #dbg_value(i8 %fb, !71, !DIExpression(), !73)
    #dbg_value(i8 %fb, !72, !DIExpression(), !73)
  call void @ext(), !dbg !74
  store i8 %fb, ptr %p, align 1, !dbg !74
  ret void, !dbg !74
}

define void @inlined(ptr %p, i1 zeroext %a) !dbg !80 {
entry:
  %fa = zext i1 %a to i8
    #dbg_value(i8 %fa, !81, !DIExpression(), !84)
  call void @ext(), !dbg !83
  store i8 %fa, ptr %p, align 1, !dbg !83
  ret void, !dbg !83
}

define void @fragment(ptr %p, i64 %s) !dbg !90 {
entry:
  %lo = trunc i64 %s to i32
    #dbg_value(i32 %lo, !91, !DIExpression(DW_OP_LLVM_fragment, 0, 32), !93)
  call void @ext(), !dbg !94
  store i32 %lo, ptr %p, align 4, !dbg !94
  ret void, !dbg !94
}

define void @not_a_copy(ptr %p, i32 %a) !dbg !100 {
entry:
  %x = add i32 %a, 1
    #dbg_value(i32 %x, !101, !DIExpression(), !103)
  call void @ext(), !dbg !104
  store i32 %x, ptr %p, align 4, !dbg !104
  ret void, !dbg !104
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "")
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !DIBasicType(name: "_Bool", size: 8, encoding: DW_ATE_boolean)
!6 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10, !5}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!7 = distinct !DISubprogram(name: "zext_bool", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!12 = !{!13}
!13 = !DILocalVariable(name: "lazy", arg: 2, scope: !7, file: !1, line: 2, type: !5)
!15 = !DILocation(line: 0, scope: !7)
!16 = !DILocation(line: 2, column: 40, scope: !7)

!20 = distinct !DISubprogram(name: "zext_bool_complex", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!21 = !DILocalVariable(name: "complex", arg: 2, scope: !20, file: !1, line: 3, type: !5)
!22 = !DILocation(line: 0, scope: !20)
!23 = !DILocation(line: 3, column: 40, scope: !20)

!30 = distinct !DISubprogram(name: "later_location", scope: !1, file: !1, line: 4, type: !8, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!31 = !DILocalVariable(name: "later", arg: 1, scope: !30, file: !1, line: 4, type: !11)
!32 = !DILocation(line: 0, scope: !30)
!33 = !DILocation(line: 4, column: 40, scope: !30)

!40 = distinct !DISubprogram(name: "local_var", scope: !1, file: !1, line: 5, type: !8, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!41 = !DILocalVariable(name: "local", scope: !40, file: !1, line: 5, type: !5)
!42 = !DILocation(line: 0, scope: !40)
!43 = !DILocation(line: 5, column: 40, scope: !40)

!50 = distinct !DISubprogram(name: "other_param", scope: !1, file: !1, line: 6, type: !8, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!51 = !DILocalVariable(name: "a", arg: 1, scope: !50, file: !1, line: 6, type: !11)
!52 = !DILocalVariable(name: "b", arg: 2, scope: !50, file: !1, line: 6, type: !55)
!53 = !DILocation(line: 0, scope: !50)
!54 = !DILocation(line: 6, column: 40, scope: !50)
!55 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)

!60 = distinct !DISubprogram(name: "two_bools", scope: !1, file: !1, line: 7, type: !8, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!61 = !DILocalVariable(name: "ta", arg: 2, scope: !60, file: !1, line: 7, type: !5)
!62 = !DILocalVariable(name: "tb", arg: 3, scope: !60, file: !1, line: 7, type: !5)
!63 = !DILocation(line: 0, scope: !60)
!64 = !DILocation(line: 7, column: 40, scope: !60)
!70 = distinct !DISubprogram(name: "shared_reg", scope: !1, file: !1, line: 8, type: !8, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!71 = !DILocalVariable(name: "sa", arg: 2, scope: !70, file: !1, line: 8, type: !5)
!72 = !DILocalVariable(name: "sb", arg: 3, scope: !70, file: !1, line: 8, type: !5)
!73 = !DILocation(line: 0, scope: !70)
!74 = !DILocation(line: 8, column: 40, scope: !70)
!80 = distinct !DISubprogram(name: "inlined", scope: !1, file: !1, line: 9, type: !8, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!82 = distinct !DISubprogram(name: "callee", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!81 = !DILocalVariable(name: "inl", arg: 1, scope: !82, file: !1, line: 1, type: !5)
!83 = !DILocation(line: 9, column: 40, scope: !80)
!84 = !DILocation(line: 0, scope: !82, inlinedAt: !83)
!90 = distinct !DISubprogram(name: "fragment", scope: !1, file: !1, line: 10, type: !8, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!91 = !DILocalVariable(name: "frag", arg: 2, scope: !90, file: !1, line: 10, type: !55)
!93 = !DILocation(line: 0, scope: !90)
!94 = !DILocation(line: 10, column: 40, scope: !90)
!100 = distinct !DISubprogram(name: "not_a_copy", scope: !1, file: !1, line: 11, type: !8, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!101 = !DILocalVariable(name: "nc", arg: 2, scope: !100, file: !1, line: 11, type: !11)
!103 = !DILocation(line: 0, scope: !100)
!104 = !DILocation(line: 11, column: 40, scope: !100)
