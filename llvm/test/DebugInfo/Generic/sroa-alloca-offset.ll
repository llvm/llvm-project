; RUN: opt %s -passes=sroa -S | FileCheck %s --check-prefixes=COMMON,OLD
; RUN: opt %s -passes=declare-to-assign,sroa -S | FileCheck %s --check-prefixes=COMMON,NEW

;; C++17 source:
;; struct two { int a, b; } gt;
;; int fun1() {
;;   auto [x, y] = gt;
;;   return x + y;
;; }
;;
;; struct four { two a, b; } gf;
;; int fun2() {
;;   auto [x, y] = gf;
;;   return x.a + y.b;
;; }
;; Plus some hand-written IR.
;;
;; Check that SROA understands how to split dbg.declares and dbg.assigns with
;; offsets into their storge (e.g., the second variable in a structured binding
;; is stored at an offset into the shared alloca).
;;
;; Additional notes:
;; We expect the same dbg.value intrinsics to come out of SROA whether assignment
;; tracking is enabled or not. However, the order of the debug intrinsics may
;; differ, and assignment tracking replaces some dbg.declares with dbg.assigns.
;;
;; Structured bindings produce an artificial variable that covers the entire
;; alloca. Although they add clutter to the test, they've been preserved in
;; order to increase coverage. These use the placehold name 'A' in comments and
;; checks.

%struct.two = type { i32, i32 }
%struct.four = type { %struct.two, %struct.two }

@gt = dso_local global %struct.two zeroinitializer, align 4, !dbg !0
@gf = dso_local global %struct.four zeroinitializer, align 4, !dbg !5


; COMMON-LABEL: @_Z4fun1v
; COMMON-NEXT: entry
;; 32 bit variable x (!27): value a_reg.
;;
;; 32 bit variable y (!28): value b_reg.
;;
;; 64 bit variable A (!29) bits [0,  32): value a_reg.
;; 64 bit variable A (!29) bits [32, 64): value b_reg.

; OLD-NEXT: %[[a_reg:.*]] = load i32, ptr @gt
; OLD-NEXT: #dbg_value(i32 %[[a_reg]], ![[x0:[0-9]+]], !DIExpression(),
; OLD-NEXT: #dbg_value(i32 %[[a_reg]], ![[A0:[0-9]+]], !DIExpression(DW_OP_LLVM_fragment, 0, 32),
; OLD-NEXT: %[[b_reg:.*]] = load i32, ptr getelementptr inbounds (i8, ptr @gt, i64 4)
; OLD-NEXT: #dbg_value(i32 %[[b_reg]], ![[y0:[0-9]+]], !DIExpression(),
; OLD-NEXT: #dbg_value(i32 %[[b_reg]], ![[A0]], !DIExpression(DW_OP_LLVM_fragment, 32, 32),

; NEW-NEXT: %[[a_reg:.*]] = load i32, ptr @gt
; NEW-NEXT: %[[b_reg:.*]] = load i32, ptr getelementptr inbounds (i8, ptr @gt, i64 4)
; NEW-NEXT: #dbg_value(i32 %[[b_reg]], ![[y0:[0-9]+]], !DIExpression(),
; NEW-NEXT: #dbg_value(i32 %[[a_reg]], ![[A0:[0-9]+]], !DIExpression(DW_OP_LLVM_fragment, 0, 32),
; NEW-NEXT: #dbg_value(i32 %[[b_reg]], ![[A0]], !DIExpression(DW_OP_LLVM_fragment, 32, 32),
; NEW-NEXT: #dbg_value(i32 %[[a_reg]], ![[x0:[0-9]+]], !DIExpression(),
define dso_local noundef i32 @_Z4fun1v() #0 !dbg !23 {
entry:
  %0 = alloca %struct.two, align 4
    #dbg_declare(ptr %0, !27, !DIExpression(), !31)
    #dbg_declare(ptr %0, !28, !DIExpression(DW_OP_plus_uconst, 4), !31)
    #dbg_declare(ptr %0, !29, !DIExpression(), !31)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %0, ptr align 4 @gt, i64 8, i1 false), !dbg !31
  %a = getelementptr inbounds %struct.two, ptr %0, i32 0, i32 0, !dbg !31
  %1 = load i32, ptr %a, align 4, !dbg !31
  %b = getelementptr inbounds %struct.two, ptr %0, i32 0, i32 1, !dbg !31
  %2 = load i32, ptr %b, align 4, !dbg !31
  %add = add nsw i32 %1, %2, !dbg !31
  ret i32 %add, !dbg !31
}

; COMMON-LABEL: _Z4fun2v()
; COMMON-NEXT: entry:
;; 64 bit variable x (!50) bits [0,  32): value aa_reg.
;; 64 bit variable x (!50) bits [32, 64): deref ab_ba_addr
;;
;; 64 bit variable y (!51) bits [0,  32): deref ab_ba_addr + 32
;; 64 bit variable y (!51) bits [32, 64): value bb_reg.
;;
;; 128 bit variable A (!52) bits [0,   32): value aa_reg
;; 128 bit variable A (!52) bits [32,  64): deref ab_ba_addr
;; 128 bit variable A (!52) bits [96, 128): value bb_reg
;;
;; NOTE: This 8 byte alloca contains x.b (4 bytes) and y.a (4 bytes).
; COMMON-NEXT: %[[ab_ba_addr:.*]] = alloca [8 x i8], align 4
; OLD-NEXT: #dbg_declare(ptr %[[ab_ba_addr]], ![[A1:[0-9]+]], !DIExpression(DW_OP_LLVM_fragment, 32, 64),
; OLD-NEXT: #dbg_declare(ptr %[[ab_ba_addr]], ![[y1:[0-9]+]], !DIExpression(DW_OP_plus_uconst, 4, DW_OP_LLVM_fragment, 0, 32),
; OLD-NEXT: #dbg_declare(ptr %[[ab_ba_addr]], ![[x1:[0-9]+]], !DIExpression(DW_OP_LLVM_fragment, 32, 32),
; OLD-NEXT: %[[aa_reg:.*]] = load i32, ptr @gf, align 4
; OLD-NEXT: #dbg_value(i32 %[[aa_reg]], ![[x1]], !DIExpression(DW_OP_LLVM_fragment, 0, 32),
; OLD-NEXT: #dbg_value(i32 %[[aa_reg]], ![[A1]], !DIExpression(DW_OP_LLVM_fragment, 0, 32),
; OLD-NEXT: call void @llvm.memcpy{{.*}}(ptr align 4 %[[ab_ba_addr]], ptr align 4 getelementptr inbounds (i8, ptr @gf, i64 4), i64 8, i1 false)
; OLD-NEXT: %[[bb_reg:.*]] = load i32, ptr getelementptr inbounds (i8, ptr @gf, i64 12), align 4
; OLD-NEXT: #dbg_value(i32 %[[bb_reg]], ![[y1]], !DIExpression(DW_OP_LLVM_fragment, 32, 32),
; OLD-NEXT: #dbg_value(i32 %[[bb_reg]], ![[A1]], !DIExpression(DW_OP_LLVM_fragment, 96, 32),

; NEW-NEXT: #dbg_assign(i1 undef, ![[x1:[0-9]+]], !DIExpression(DW_OP_LLVM_fragment, 32, 32), ![[#]], ptr %[[ab_ba_addr]], !DIExpression(),
; NEW-NEXT: #dbg_assign(i1 undef, ![[A1:[0-9]+]], !DIExpression(DW_OP_LLVM_fragment, 32, 64), ![[#]], ptr %[[ab_ba_addr]], !DIExpression(),
; NEW-NEXT: #dbg_declare(ptr %[[ab_ba_addr]], ![[y1:[0-9]+]], !DIExpression(DW_OP_plus_uconst, 4, DW_OP_LLVM_fragment, 0, 32),
; NEW-NEXT: %[[aa_reg:.*]] = load i32, ptr @gf, align 4
; NEW-NEXT: llvm.memcpy{{.*}}(ptr align 4 %[[ab_ba_addr]], ptr align 4 getelementptr inbounds (i8, ptr @gf, i64 4), i64 8, i1 false){{.*}}, !DIAssignID ![[ID:[0-9]+]]
; NEW-NEXT: %[[bb_reg:.*]] = load i32, ptr getelementptr inbounds (i8, ptr @gf, i64 12), align 4
; NEW-NEXT: #dbg_value(i32 %[[bb_reg]], ![[y1:[0-9]+]], !DIExpression(DW_OP_LLVM_fragment, 32, 32),
; NEW-NEXT: #dbg_value(i32 %[[aa_reg]], ![[A1]], !DIExpression(DW_OP_LLVM_fragment, 0, 32),
; NEW-NEXT: #dbg_assign(i1 undef, ![[A1]], !DIExpression(DW_OP_LLVM_fragment, 32, 64), ![[ID]], ptr %[[ab_ba_addr]], !DIExpression(),
; NEW-NEXT: #dbg_value(i32 %[[bb_reg]], ![[A1]], !DIExpression(DW_OP_LLVM_fragment, 96, 32),
; NEW-NEXT: #dbg_value(i32 %[[aa_reg]], ![[x1]], !DIExpression(DW_OP_LLVM_fragment, 0, 32),
define dso_local noundef i32 @_Z4fun2v() #0 !dbg !48 {
entry:
  %0 = alloca %struct.four, align 4
    #dbg_declare(ptr %0, !50, !DIExpression(), !54)
    #dbg_declare(ptr %0, !51, !DIExpression(DW_OP_plus_uconst, 8), !54)
    #dbg_declare(ptr %0, !52, !DIExpression(), !54)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %0, ptr align 4 @gf, i64 16, i1 false), !dbg !54
  %a = getelementptr inbounds %struct.four, ptr %0, i32 0, i32 0, !dbg !54
  %a1 = getelementptr inbounds %struct.two, ptr %a, i32 0, i32 0, !dbg !54
  %1 = load i32, ptr %a1, align 4, !dbg !54
  %b = getelementptr inbounds %struct.four, ptr %0, i32 0, i32 1, !dbg !54
  %b2 = getelementptr inbounds %struct.two, ptr %b, i32 0, i32 1, !dbg !54
  %2 = load i32, ptr %b2, align 4, !dbg !54
  %add = add nsw i32 %1, %2, !dbg !54
  ret i32 %add, !dbg !54
}

;; Hand-written part to test what happens when variables are smaller than the
;; new alloca slices (i.e., check offset rewriting works correctly). Note that
;; mem2reg incorrectly preserves the offest in the DIExpression of a variable
;; stuffed into the upper bits of a value (that is a bug), e.g. alloca+offset
;; becomes vreg+offest. It should either convert the offest to a shift, encode
;; the register-bit offest using DW_OP_bit_piece, or use the new
;; DW_OP_LLVM_extract_bits_[sz]ext operation.
; COMMON-LABEL: _Z4fun3v()
; COMMON-NEXT: entry:
;; 16 bit variable e (!61): value ve (upper bits)
;;
;; 16 bit variable f (!62): value vgf (lower bits)
;; 16 bit variable g (!63): value vgf (upper bits)
;;
;; 16 bit variable h (!64): deref dead_64_128
; COMMON-NEXT: %[[dead_64_128:.*]] = alloca %struct.two
; COMMON-NEXT: #dbg_declare(ptr %[[dead_64_128]], ![[h:[0-9]+]], !DIExpression(),
; COMMON-NEXT: %[[ve:.*]] = load i32, ptr @gf
;; FIXME: mem2reg bug - offset is incorrect - see comment above.
; COMMON-NEXT: #dbg_value(i32 %[[ve]], ![[e:[0-9]+]], !DIExpression(DW_OP_plus_uconst, 2),
; COMMON-NEXT: %[[vfg:.*]] = load i32, ptr getelementptr inbounds (i8, ptr @gf, i64 4)
; COMMON-NEXT: #dbg_value(i32 %[[vfg]], ![[f:[0-9]+]], !DIExpression(),
;; FIXME: mem2reg bug - offset is incorrect - see comment above.
; COMMON-NEXT: #dbg_value(i32 %[[vfg]], ![[g:[0-9]+]], !DIExpression(DW_OP_plus_uconst, 2),
define dso_local noundef i32 @_Z4fun3v() #0 !dbg !55 {
entry:
  %0 = alloca %struct.four, align 4
    #dbg_declare(ptr %0, !61, !DIExpression(DW_OP_plus_uconst, 2), !58)
    #dbg_declare(ptr %0, !62, !DIExpression(DW_OP_plus_uconst, 4), !58)
    #dbg_declare(ptr %0, !63, !DIExpression(DW_OP_plus_uconst, 6), !58)
    #dbg_declare(ptr %0, !64, !DIExpression(DW_OP_plus_uconst, 8), !58)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %0, ptr align 4 @gf, i64 16, i1 false), !dbg !58
  %1 = getelementptr inbounds %struct.four, ptr %0, i32 0, i32 0, !dbg !58
  %2 = getelementptr inbounds %struct.two, ptr %1, i32 0, i32 1, !dbg !58
  %3 = load i32, ptr %2, align 4, !dbg !58
  ret i32 %3, !dbg !58
}

;; Check that DW_OP_extract_bits_[sz]ext compose with expression offsets and
;; that new fragments are not created. DW_OP_extract_bits_[sz]ext and fragments
;; don't compose currently (but could). There are checks that expressions with
;; bit extracts and fragments are dropped in SROA the test
;; in llvm/test/DebugInfo/Generic/sroa-extract-bits.ll. FIXME: Don't do that.
;;
;; Checks are inline for this one.
;;
;; %p alloca is 128 bits
;; SROA is going to split it in half, discard the lower bits, then split
;; the upper bits in half and discard the upper bits leaving us with
;; bits [64, 96) of the original alloca.
;;
; COMMON-LABEL: fun4
define dso_local noundef i32 @fun4(i64 %0) !dbg !65 {
entry:
  %p = alloca [2 x i64]
  %1 = getelementptr inbounds [2 x i64], ptr %p, i32 0, i32 1
  store i64 %0, ptr %1
  ; COMMON: %p.sroa.0.8.extract.trunc = trunc i64 %0 to i32
  ;; Simple case - the expression offset (8 bytes) matches the offset of the
  ;; slice into the alloca, so can be discarded away entirely.
  ; COMMON-NEXT: #dbg_value(i32 %p.sroa.0.8.extract.trunc, ![[p:[0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 32)
    #dbg_declare(ptr %p, !67, !DIExpression(DW_OP_plus_uconst, 8, DW_OP_LLVM_extract_bits_zext, 0, 32), !66)
  ;; The expression offset is 6 bytes, with a bit-extract offset of 32 bits from
  ;; there for a total offset of 80 bits. SROA is going to split the alloca in
  ;; half (at bit 64). The new expression needs a final bit extract offset of
  ;; 80-64=16 bits applied to the mem2reg'd value.
  ; COMMON-NEXT: #dbg_value(i32 %p.sroa.0.8.extract.trunc, ![[q:[0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 16, 8)
    #dbg_declare(ptr %p, !68, !DIExpression(DW_OP_plus_uconst, 6, DW_OP_LLVM_extract_bits_zext, 32, 8), !66)
  ;; FIXME: Just as in _Z4fun3v, the offset from the new alloca (2 bytes) is
  ;; correct but mem2reg needs to change it from an offset to a shift or
  ;; adjust the bit-extract (e.g., add the 2 byte offset to the existing 8 bit
  ;; offset for a 24 bit total bit-extract offset).
  ; COMMON-NEXT: #dbg_value(i32 %p.sroa.0.8.extract.trunc, ![[r:[0-9]+]], !DIExpression(DW_OP_plus_uconst, 2, DW_OP_LLVM_extract_bits_zext, 8, 8)
    #dbg_declare(ptr %p, !69, !DIExpression(DW_OP_plus_uconst, 10, DW_OP_LLVM_extract_bits_zext, 8, 8), !66)
  %2 = load i32, ptr %1, align 4
  ret i32 %2
}

; COMMON-DAG: ![[x0]] = !DILocalVariable(name: "x",
; COMMON-DAG: ![[y0]] = !DILocalVariable(name: "y",
; COMMON-DAG: ![[A0]] = !DILocalVariable(scope:

; COMMON-DAG: ![[x1]] = !DILocalVariable(name: "x",
; COMMON-DAG: ![[y1]] = !DILocalVariable(name: "y",
; COMMON-DAG: ![[A1]] = !DILocalVariable(scope:

; COMMON-DAG: ![[e]] = !DILocalVariable(name: "e",
; COMMON-DAG: ![[f]] = !DILocalVariable(name: "f",
; COMMON-DAG: ![[g]] = !DILocalVariable(name: "g",
; COMMON-DAG: ![[h]] = !DILocalVariable(name: "h",

; COMMON-DAG: ![[p]] = !DILocalVariable(name: "p"
; COMMON-DAG: ![[q]] = !DILocalVariable(name: "q"
; COMMON-DAG: ![[r]] = !DILocalVariable(name: "r"

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!16, !17}
!llvm.ident = !{!22}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "gt", scope: !2, file: !3, line: 1, type: !10, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 17.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.cpp", directory: "/")
!4 = !{!0, !5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "gf", scope: !2, file: !3, line: 7, type: !7, isLocal: false, isDefinition: true)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "four", file: !3, line: 7, size: 128, flags: DIFlagTypePassByValue, elements: !8, identifier: "_ZTS4four")
!8 = !{!9, !15}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !7, file: !3, line: 7, baseType: !10, size: 64)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "two", file: !3, line: 1, size: 64, flags: DIFlagTypePassByValue, elements: !11, identifier: "_ZTS3two")
!11 = !{!12, !14}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !10, file: !3, line: 1, baseType: !13, size: 32)
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !10, file: !3, line: 1, baseType: !13, size: 32, offset: 32)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !7, file: !3, line: 7, baseType: !10, size: 64, offset: 64)
!16 = !{i32 7, !"Dwarf Version", i32 5}
!17 = !{i32 2, !"Debug Info Version", i32 3}
!22 = !{!"clang version 17.0.0"}
!23 = distinct !DISubprogram(name: "fun1", linkageName: "_Z4fun1v", scope: !3, file: !3, line: 2, type: !24, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !26)
!24 = !DISubroutineType(types: !25)
!25 = !{!13}
!26 = !{!27, !28, !29}
!27 = !DILocalVariable(name: "x", scope: !23, file: !3, line: 3, type: !13)
!28 = !DILocalVariable(name: "y", scope: !23, file: !3, line: 3, type: !13)
!29 = !DILocalVariable(scope: !23, file: !3, line: 3, type: !10)
!31 = !DILocation(line: 3, column: 9, scope: !23)
!48 = distinct !DISubprogram(name: "fun2", linkageName: "_Z4fun2v", scope: !3, file: !3, line: 8, type: !24, scopeLine: 8, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !49)
!49 = !{!50, !51, !52}
!50 = !DILocalVariable(name: "x", scope: !48, file: !3, line: 9, type: !10)
!51 = !DILocalVariable(name: "y", scope: !48, file: !3, line: 9, type: !10)
!52 = !DILocalVariable(scope: !48, file: !3, line: 9, type: !7)
!54 = !DILocation(line: 9, column: 9, scope: !48)
!55 = distinct !DISubprogram(name: "fun3", linkageName: "_Z4fun3v", scope: !3, file: !3, line: 8, type: !24, scopeLine: 8, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !56)
!56 = !{}
!58 = !DILocation(line: 9, column: 9, scope: !55)
!60 = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
!61 = !DILocalVariable(name: "e", scope: !55, file: !3, line: 9, type: !60)
!62 = !DILocalVariable(name: "f", scope: !55, file: !3, line: 9, type: !60)
!63 = !DILocalVariable(name: "g", scope: !55, file: !3, line: 9, type: !60)
!64 = !DILocalVariable(name: "h", scope: !55, file: !3, line: 9, type: !60)
!65 = distinct !DISubprogram(name: "fun4", linkageName: "_Z4fun4v", scope: !3, file: !3, line: 8, type: !24, scopeLine: 8, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !56)
!66 = !DILocation(line: 9, column: 9, scope: !65)
!67 = !DILocalVariable(name: "p", scope: !65, file: !3, line: 9, type: !13)
!68 = !DILocalVariable(name: "q", scope: !65, file: !3, line: 9, type: !13)
!69 = !DILocalVariable(name: "r", scope: !65, file: !3, line: 9, type: !13)
