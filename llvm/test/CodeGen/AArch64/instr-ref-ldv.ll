; RUN: llc -O2 -experimental-debug-variable-locations %s -stop-after=livedebugvalues -mtriple=arm64-apple-macosx15.0.0 -o - | FileCheck %s

; CHECK: $w{{[0-9]+}} = ORRWrs $wzr, killed $w{{[0-9]+}}, 0
; CHECK-NEXT: DBG_INSTR_REF !{{[0-9]+}}, !DIExpression(DW_OP_LLVM_arg, 0), dbg-instr-ref({{[0-9]+}}, 0), debug-location !{{[0-9]+}}

; This test makes sure that instruction referenced livedebugvalues pass doesn't crash when an ORRWrr is present before 
; aarch64-isel and is converted to an ORRWrs with a shift amount immediate value of 0 before livedebugvalues, in this 
; test case the MIR before both passes is shown below:

; Before aarch64-isel
; %11:gpr32 = ORRWrr $wzr, killed %10:gpr32, debug-location !5; :0
; %0:gpr64all = SUBREG_TO_REG 0, killed %11:gpr32, %subreg.sub_32, debug-location !5; :0
; DBG_INSTR_REF !7, !DIExpression(DW_OP_LLVM_arg, 0), %0:gpr64all, debug-location !11; :0 @[ :0 ] line no:0

; Before livedebugvalues
; $w0 = ORRWrs $wzr, killed $w3, 0
; DBG_INSTR_REF !7, !DIExpression(DW_OP_LLVM_arg, 0), dbg-instr-ref(3, 0), debug-location !11; :0 @[ :0 ] line no:0

; The livedebugvalues pass will consider the ORRWrs variant as a copy, therefore the aarch64-isel call to 
; salvageCopySSA should do the same.

%"class.llvm::iterator_range.53" = type { %"class.llvm::opt::arg_iterator.54", %"class.llvm::opt::arg_iterator.54" }
%"class.llvm::opt::arg_iterator.54" = type { %"class.std::__1::reverse_iterator", %"class.std::__1::reverse_iterator", [2 x %"class.llvm::opt::OptSpecifier"] }
%"class.std::__1::reverse_iterator" = type { ptr, ptr }
%"class.llvm::opt::OptSpecifier" = type { i32 }
declare noundef zeroext i1 @_ZNK4llvm3opt6Option7matchesENS0_12OptSpecifierE(ptr noundef nonnull align 8 dereferenceable(16), i64) local_unnamed_addr #1
define noundef zeroext i1 @_ZNK4llvm3opt7ArgList14hasFlagNoClaimENS0_12OptSpecifierES2_b(ptr noundef nonnull align 8 dereferenceable(184) %this, i64 %Pos.coerce, i64 %Neg.coerce, i1 noundef zeroext %Default) local_unnamed_addr #2 !dbg !9383 {
entry:
  %ref.tmp.i = alloca %"class.llvm::iterator_range.53", align 8
  %coerce.val.ii6 = and i64 %Pos.coerce, 4294967295, !dbg !9393
    #dbg_value(i64 %coerce.val.ii6, !9452, !DIExpression(), !9480)
  %__begin0.sroa.4.0.ref.tmp.sroa_idx.i = getelementptr inbounds i8, ptr %ref.tmp.i, i64 8, !dbg !9480
  %__begin0.sroa.4.0.copyload.i = load ptr, ptr %__begin0.sroa.4.0.ref.tmp.sroa_idx.i, align 8, !dbg !9480
  %__end0.sroa.4.0.end_iterator.i.sroa_idx.i = getelementptr inbounds i8, ptr %ref.tmp.i, i64 48, !dbg !9480
  %__end0.sroa.4.0.copyload.i = load ptr, ptr %__end0.sroa.4.0.end_iterator.i.sroa_idx.i, align 8, !dbg !9480
  %cmp.i.i.i.not.i = icmp eq ptr %__begin0.sroa.4.0.copyload.i, %__end0.sroa.4.0.copyload.i, !dbg !9480
  br i1 %cmp.i.i.i.not.i, label %_ZNK4llvm3opt7ArgList17getLastArgNoClaimIJNS0_12OptSpecifierES3_EEEPNS0_3ArgEDpT_.exit.thread, label %_ZNK4llvm3opt7ArgList17getLastArgNoClaimIJNS0_12OptSpecifierES3_EEEPNS0_3ArgEDpT_.exit, !dbg !9480
_ZNK4llvm3opt7ArgList17getLastArgNoClaimIJNS0_12OptSpecifierES3_EEEPNS0_3ArgEDpT_.exit.thread: ; preds = %entry
  br label %1, !dbg !9480
_ZNK4llvm3opt7ArgList17getLastArgNoClaimIJNS0_12OptSpecifierES3_EEEPNS0_3ArgEDpT_.exit: ; preds = %entry
  %incdec.ptr.i.i.i = getelementptr inbounds i8, ptr %__begin0.sroa.4.0.copyload.i, i64 -8, !dbg !9480
  %0 = load ptr, ptr %incdec.ptr.i.i.i, align 8, !dbg !9527, !tbaa !9528
  %tobool.not.not = icmp eq ptr %0, null, !dbg !9480
  br i1 %tobool.not.not, label %1, label %cleanup, !dbg !9480
cleanup:                                          ; preds = %_ZNK4llvm3opt7ArgList17getLastArgNoClaimIJNS0_12OptSpecifierES3_EEEPNS0_3ArgEDpT_.exit
  %call13 = call noundef zeroext i1 @_ZNK4llvm3opt6Option7matchesENS0_12OptSpecifierE(ptr noundef nonnull align 8 dereferenceable(16) %0, i64 %coerce.val.ii6) #3, !dbg !9480
  br label %1
  %2 = phi i1 [ %call13, %cleanup ], [ %Default, %_ZNK4llvm3opt7ArgList17getLastArgNoClaimIJNS0_12OptSpecifierES3_EEEPNS0_3ArgEDpT_.exit ], [ %Default, %_ZNK4llvm3opt7ArgList17getLastArgNoClaimIJNS0_12OptSpecifierES3_EEEPNS0_3ArgEDpT_.exit.thread ]
  ret i1 %2, !dbg !9480
}
!llvm.module.flags = !{!2, !6}
!llvm.dbg.cu = !{!7}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 7, !"frame-pointer", i32 1}
!7 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !8, emissionKind: FullDebug, sdk: "MacOSX15.3.sdk")
!8 = !DIFile(filename: "/Users/shubhamrastogi/Development/llvm-project-instr-ref/llvm-project/llvm/lib/Option/ArgList.cpp", directory: "/Users/shubhamrastogi/Development/llvm-project-instr-ref/llvm-project/build-instr-ref-stage2", checksumkind: CSK_MD5, checksum: "a3198e8ace679c7b1581a26b5583c658")
!3116 = distinct !DICompositeType(tag: DW_TAG_class_type, size: 32)
!9383 = distinct !DISubprogram(unit: !7, flags: DIFlagArtificial | DIFlagObjectPointer)
!9391 = distinct !DILexicalBlock(scope: !9383, line: 80, column: 12)
!9393 = !DILocation(scope: !9391)
!9440 = distinct !DILexicalBlock(scope: !9441, line: 269, column: 5)
!9441 = distinct !DILexicalBlock(scope: !9442, line: 269, column: 5)
!9442 = distinct !DISubprogram(unit: !7, retainedNodes: !9450)
!9450 = !{}
!9452 = !DILocalVariable(scope: !9442, type: !3116)
!9478 = distinct !DILocation(scope: !9391)
!9480 = !DILocation(scope: !9441, inlinedAt: !9478)
!9527 = !DILocation(scope: !9440, inlinedAt: !9478)
!9528 = !{!"any pointer", !9530, i64 0}
!9530 = !{}
