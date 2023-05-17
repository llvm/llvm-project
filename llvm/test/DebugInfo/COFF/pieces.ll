; RUN: llc < %s -experimental-debug-variable-locations=true | FileCheck %s --check-prefix=ASM
; RUN: llc < %s -filetype=obj -experimental-debug-variable-locations=true | llvm-readobj --codeview - | FileCheck %s --check-prefix=OBJ

; Compile with -O1 as C

; struct IntPair { int x, y; };
; struct PadRight { long a; int b; };
; struct PadLeft { int a; long b; };
; struct Nested { struct PadLeft a[2]; };
;
; extern int g(int r);
; extern int i;
; extern int n;
;
; int loop_csr() {
;   struct IntPair o = {0, 0};
;   for (i = 0; i < n; i++) {
;     o.x = g(o.x);
;     o.y = g(o.y);
;   }
;   return o.x + o.y;
; }
;
; int pad_right(struct PadRight o) {
;   return o.b;
; }
;
; int pad_left(struct PadLeft o) {
;   return o.a;
; }
;
; int nested(struct Nested o) {
;   struct PadLeft p = o.a[1];
;   return p.b;
; }

; ASM-LABEL: loop_csr: # @loop_csr
; ASM:        #DEBUG_VALUE: loop_csr:o <- [DW_OP_LLVM_fragment 0 32] 0
; ASM:        #DEBUG_VALUE: loop_csr:o <- [DW_OP_LLVM_fragment 32 32] 0
; ASM: # %bb.2:                                 # %for.body.preheader
; ASM:         xorl    %edi, %edi
; ASM:         xorl    %esi, %esi
; ASM: [[oy_ox_start:\.Ltmp[0-9]+]]:
; ASM:         .p2align        4, 0x90
; ASM: .LBB0_3:                                # %for.body
; ASM:        #DEBUG_VALUE: loop_csr:o <- [DW_OP_LLVM_fragment 0 32] $edi
; ASM:        #DEBUG_VALUE: loop_csr:o <- [DW_OP_LLVM_fragment 32 32] $esi
; ASM:        .cv_loc 0 1 13 11               # t.c:13:11
; ASM:        movl    %edi, %ecx
; ASM:        callq   g
; ASM:        movl    %eax, %edi
; ASM: [[ox_start:\.Ltmp[0-9]+]]:
; ASM:         #DEBUG_VALUE: loop_csr:o <- [DW_OP_LLVM_fragment 0 32] $edi
; ASM:         .cv_loc 0 1 14 11               # t.c:14:11
; ASM:         movl    %esi, %ecx
; ASM:         callq   g
; ASM:         movl    %eax, %esi
; ASM: [[oy_start:\.Ltmp[0-9]+]]:
; ASM:         #DEBUG_VALUE: loop_csr:o <- [DW_OP_LLVM_fragment 32 32] $esi
; ASM:         cmpl    n(%rip), %eax
; ASM:         jl      .LBB0_3
; ASM: [[loopskip_start:\.Ltmp[0-9]+]]:
; ASM:         #DEBUG_VALUE: loop_csr:o <- [DW_OP_LLVM_fragment 0 32] 0
; ASM:         xorl    %esi, %esi
; ASM:         xorl    %edi, %edi
; ASM: [[oy_end:\.Ltmp[0-9]+]]:
; ASM:         addl    %edi, %esi
; ASM:         movl    %esi, %eax

; XXX FIXME: the debug value line after loopskip_start should be repeated
; because both fields of 'o' are zero flowing into this block. However, it
; appears livedebugvalues doesn't account for fragments.

; ASM-LABEL: pad_right: # @pad_right
; ASM:         movq    %rcx, %rax
; ASM: [[pad_right_tmp:\.Ltmp[0-9]+]]:
; ASM:         #DEBUG_VALUE: pad_right:o <- [DW_OP_LLVM_fragment 32 32] $eax
; ASM:         retq
; ASM: [[pad_right_end:\.Lfunc_end1]]:


; ASM-LABEL: pad_left: # @pad_left
; ASM:         .cv_loc 2 1 24 3                # t.c:24:3
; ASM:         movq    %rcx, %rax
; ASM: [[pad_left_tmp:\.Ltmp[0-9]+]]:
; ASM:         #DEBUG_VALUE: pad_left:o <- [DW_OP_LLVM_fragment 0 32] $ecx
; ASM:         retq
; ASM: [[pad_left_end:\.Lfunc_end2]]:


; ASM-LABEL: nested: # @nested
; ASM:         #DEBUG_VALUE: nested:o <- [DW_OP_deref] [$rcx+0]
; ASM:         movl    12(%rcx), %eax
; ASM: [[p_start:\.Ltmp[0-9]+]]:
; ASM:         #DEBUG_VALUE: nested:p <- [DW_OP_LLVM_fragment 32 32] $eax
; ASM:         retq

; ASM-LABEL: bitpiece_spill: # @bitpiece_spill
; ASM:         #DEBUG_VALUE: bitpiece_spill:o <- [DW_OP_LLVM_fragment 0 32] 0
; ASM:         xorl    %ecx, %ecx
; ASM:         callq   g
; ASM:         movl    %eax, [[offset_o_x:[0-9]+]](%rsp)          # 4-byte Spill
; ASM: [[spill_o_x_start:\.Ltmp[0-9]+]]:
; ASM:         #DEBUG_VALUE: bitpiece_spill:o <- [DW_OP_plus_uconst [[offset_o_x]], DW_OP_deref, DW_OP_LLVM_fragment 32 32] $rsp
; ASM:         #APP
; ASM:         #NO_APP
; ASM:         movl    [[offset_o_x]](%rsp), %eax          # 4-byte Reload
; ASM:         retq
; ASM: [[spill_o_x_end_func:\.Ltmp[0-9]+]]:
; ASM-NEXT: .Lfunc_end4:


; ASM-LABEL:  .short  4423                    # Record kind: S_GPROC32_ID
; ASM:        .asciz  "loop_csr"              # Function name
; ASM:        .short	4359                    # Record kind: S_CONSTANT
; ASM:      	.long	4099                      # Type
; ASM:        .byte	0x00, 0x00                # Value
; ASM:        .asciz  "o"


; OBJ-LABEL: GlobalProcIdSym {
; OBJ:         Kind: S_GPROC32_ID (0x1147)
; OBJ:         DisplayName: loop_csr
; OBJ:       }
; OBJ:       ConstantSym {
; OBJ:         Kind: S_CONSTANT (0x1107)
; OBJ:         Type: IntPair (0x1003)
; OBJ:         Value: 0
; OBJ:         Name: o
; OBJ:       }
; OBJ:       ProcEnd {
; OBJ:       }

; ASM-LABEL:  .short  4423                    # Record kind: S_GPROC32_ID
; ASM:        .asciz  "pad_right"             # Function name
; ASM:        .short  4414                    # Record kind: S_LOCAL
; ASM:        .asciz  "o"
; ASM:        .cv_def_range    [[pad_right_tmp]] [[pad_right_end]], subfield_reg, 17, 4

; OBJ-LABEL: GlobalProcIdSym {
; OBJ:         Kind: S_GPROC32_ID (0x1147)
; OBJ:         DisplayName: pad_right
; OBJ:       }
; OBJ:       LocalSym {
; OBJ:         VarName: o
; OBJ:       }
; OBJ:       DefRangeSubfieldRegisterSym {
; OBJ:         Register: EAX (0x11)
; OBJ:         MayHaveNoName: 0
; OBJ:         OffsetInParent: 4
; OBJ:         LocalVariableAddrRange {
; OBJ:         }
; OBJ:       }
; OBJ:       ProcEnd {
; OBJ:       }

; ASM-LABEL:  .short  4423                    # Record kind: S_GPROC32_ID
; ASM:        .asciz  "pad_left"              # Function name
; ASM:        .short  4414                    # Record kind: S_LOCAL
; ASM:        .asciz  "o"
; ASM:        .cv_def_range    [[pad_left_tmp]] [[pad_left_end]], subfield_reg, 18, 0

; OBJ-LABEL: GlobalProcIdSym {
; OBJ:         Kind: S_GPROC32_ID (0x1147)
; OBJ:         DisplayName: pad_left
; OBJ:       }
; OBJ:       LocalSym {
; OBJ:         VarName: o
; OBJ:       }
; OBJ:       DefRangeSubfieldRegisterSym {
; OBJ:         Register: ECX (0x12)
; OBJ:         MayHaveNoName: 0
; OBJ:         OffsetInParent: 0
; OBJ:         LocalVariableAddrRange {
; OBJ:         }
; OBJ:       }
; OBJ:       ProcEnd {
; OBJ:       }

; ASM-LABEL:  .short  4423                    # Record kind: S_GPROC32_ID
; ASM:        .asciz  "nested"                # Function name
; ASM:        .short  4414                    # Record kind: S_LOCAL
; ASM:        .asciz  "o"
; ASM:        .cv_def_range    .Lfunc_begin3 .Lfunc_end3, reg_rel, 330, 0, 0
; ASM:        .short  4414                    # Record kind: S_LOCAL
; ASM:        .asciz  "p"
; ASM:        .cv_def_range    [[p_start]] .Lfunc_end3, subfield_reg, 17, 4

; OBJ-LABEL: GlobalProcIdSym {
; OBJ:         Kind: S_GPROC32_ID (0x1147)
; OBJ:         DisplayName: nested
; OBJ:       }
; OBJ:       LocalSym {
; OBJ:         Type: Nested&
; OBJ:         VarName: o
; OBJ:       }
; OBJ:       DefRangeRegisterRelSym {
; OBJ:         BaseRegister: RCX (0x14A)
; OBJ:         HasSpilledUDTMember: No
; OBJ:         OffsetInParent: 0
; OBJ:         BasePointerOffset: 0
; OBJ:         LocalVariableAddrRange {
; OBJ:         }
; OBJ:       }
; OBJ:       LocalSym {
; OBJ:         VarName: p
; OBJ:       }
; OBJ:       DefRangeSubfieldRegisterSym {
; OBJ:         Register: EAX (0x11)
; OBJ:         MayHaveNoName: 0
; OBJ:         OffsetInParent: 4
; OBJ:         LocalVariableAddrRange {
; OBJ:         }
; OBJ:       }
; OBJ:       ProcEnd {
; OBJ:       }


; ASM-LABEL:  .short  4423                    # Record kind: S_GPROC32_ID
; ASM:        .asciz  "bitpiece_spill"        # Function name
; ASM:      	.short	4359                    # Record kind: S_CONSTANT
; ASM:      	.long	4099                      # Type
; ASM:      	.byte	0x00, 0x00                # Value
; ASM:        .asciz  "o"                     # Name

; OBJ-LABEL: GlobalProcIdSym {
; OBJ:         Kind: S_GPROC32_ID (0x1147)
; OBJ:         DisplayName: bitpiece_spill
; OBJ:       }
; OBJ:       ConstantSym {
; OBJ:         Kind: S_CONSTANT (0x1107)
; OBJ:         Type: IntPair (0x1003)
; OBJ:         Value: 0
; OBJ:         Name: o
; OBJ:       }
; OBJ:       ProcEnd {
; OBJ:       }



; ModuleID = 't.c'
source_filename = "t.c"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24210"

%struct.IntPair = type { i32, i32 }
%struct.PadRight = type { i32, i32 }
%struct.PadLeft = type { i32, i32 }
%struct.Nested = type { [2 x %struct.PadLeft] }

@i = external local_unnamed_addr global i32, align 4
@n = external local_unnamed_addr global i32, align 4

; Function Attrs: nounwind uwtable
define i32 @loop_csr() local_unnamed_addr #0 !dbg !7 {
entry:
  tail call void @llvm.dbg.declare(metadata ptr undef, metadata !12, metadata !17), !dbg !18
  tail call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !19), !dbg !18
  tail call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !20), !dbg !18
  tail call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !19), !dbg !18
  tail call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !20), !dbg !18
  store i32 0, ptr @i, align 4, !dbg !21, !tbaa !24
  %0 = load i32, ptr @n, align 4, !dbg !28, !tbaa !24
  %cmp9 = icmp sgt i32 %0, 0, !dbg !29
  br i1 %cmp9, label %for.body, label %for.end, !dbg !30

for.body:                                         ; preds = %entry, %for.body
  %o.sroa.0.011 = phi i32 [ %call, %for.body ], [ 0, %entry ]
  %o.sroa.5.010 = phi i32 [ %call2, %for.body ], [ 0, %entry ]
  tail call void @llvm.dbg.value(metadata i32 %o.sroa.0.011, metadata !12, metadata !19), !dbg !18
  tail call void @llvm.dbg.value(metadata i32 %o.sroa.5.010, metadata !12, metadata !20), !dbg !18
  %call = tail call i32 @g(i32 %o.sroa.0.011) #5, !dbg !31
  tail call void @llvm.dbg.value(metadata i32 %call, metadata !12, metadata !19), !dbg !18
  %call2 = tail call i32 @g(i32 %o.sroa.5.010) #5, !dbg !33
  tail call void @llvm.dbg.value(metadata i32 %call2, metadata !12, metadata !20), !dbg !18
  %1 = load i32, ptr @i, align 4, !dbg !21, !tbaa !24
  %inc = add nsw i32 %1, 1, !dbg !21
  store i32 %inc, ptr @i, align 4, !dbg !21, !tbaa !24
  %2 = load i32, ptr @n, align 4, !dbg !28, !tbaa !24
  %cmp = icmp slt i32 %inc, %2, !dbg !29
  br i1 %cmp, label %for.body, label %for.end, !dbg !30, !llvm.loop !34

for.end:                                          ; preds = %for.body, %entry
  %o.sroa.5.0.lcssa = phi i32 [ 0, %entry ], [ %call2, %for.body ]
  %o.sroa.0.0.lcssa = phi i32 [ 0, %entry ], [ %call, %for.body ]
  %add = add nsw i32 %o.sroa.0.0.lcssa, %o.sroa.5.0.lcssa, !dbg !36
  ret i32 %add, !dbg !37
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare i32 @g(i32) local_unnamed_addr #2

; Function Attrs: nounwind readnone uwtable
define i32 @pad_right(i64 %o.coerce) local_unnamed_addr #3 !dbg !38 {
entry:
  %o.sroa.1.0.extract.shift = lshr i64 %o.coerce, 32
  %o.sroa.1.0.extract.trunc = trunc i64 %o.sroa.1.0.extract.shift to i32
  tail call void @llvm.dbg.value(metadata i32 %o.sroa.1.0.extract.trunc, metadata !47, metadata !20), !dbg !48
  tail call void @llvm.dbg.declare(metadata ptr undef, metadata !47, metadata !17), !dbg !48
  ret i32 %o.sroa.1.0.extract.trunc, !dbg !49
}

; Function Attrs: nounwind readnone uwtable
define i32 @pad_left(i64 %o.coerce) local_unnamed_addr #3 !dbg !50 {
entry:
  %o.sroa.0.0.extract.trunc = trunc i64 %o.coerce to i32
  tail call void @llvm.dbg.value(metadata i32 %o.sroa.0.0.extract.trunc, metadata !58, metadata !19), !dbg !59
  tail call void @llvm.dbg.declare(metadata ptr undef, metadata !58, metadata !17), !dbg !59
  ret i32 %o.sroa.0.0.extract.trunc, !dbg !60
}

; Function Attrs: nounwind readonly uwtable
define i32 @nested(ptr nocapture readonly %o) local_unnamed_addr #4 !dbg !61 {
entry:
  tail call void @llvm.dbg.declare(metadata ptr %o, metadata !71, metadata !73), !dbg !74
  tail call void @llvm.dbg.declare(metadata ptr undef, metadata !72, metadata !17), !dbg !75
  %p.sroa.3.0..sroa_idx2 = getelementptr inbounds %struct.Nested, ptr %o, i64 0, i32 0, i64 1, i32 1, !dbg !76
  %p.sroa.3.0.copyload = load i32, ptr %p.sroa.3.0..sroa_idx2, align 4, !dbg !76
  tail call void @llvm.dbg.value(metadata i32 %p.sroa.3.0.copyload, metadata !72, metadata !20), !dbg !75
  ret i32 %p.sroa.3.0.copyload, !dbg !77
}

; Function Attrs: nounwind uwtable
define i32 @bitpiece_spill() local_unnamed_addr #0 !dbg !78 {
entry:
  tail call void @llvm.dbg.declare(metadata ptr undef, metadata !80, metadata !17), !dbg !81
  tail call void @llvm.dbg.value(metadata i32 0, metadata !80, metadata !19), !dbg !81
  %call = tail call i32 @g(i32 0) #5, !dbg !82
  tail call void @llvm.dbg.value(metadata i32 %call, metadata !80, metadata !20), !dbg !81
  tail call void asm sideeffect "", "~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{dirflag},~{fpsr},~{flags}"() #5, !dbg !83, !srcloc !84
  ret i32 %call, !dbg !85
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0.0 (trunk 283332) (llvm/trunk 283355)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.c", directory: "C:\5Csrc\5Cllvm\5Cbuild")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 4.0.0 (trunk 283332) (llvm/trunk 283355)"}
!7 = distinct !DISubprogram(name: "loop_csr", scope: !1, file: !1, line: 10, type: !8, isLocal: false, isDefinition: true, scopeLine: 10, isOptimized: true, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "o", scope: !7, file: !1, line: 11, type: !13)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "IntPair", file: !1, line: 1, size: 64, align: 32, elements: !14)
!14 = !{!15, !16}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !13, file: !1, line: 1, baseType: !10, size: 32, align: 32)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !13, file: !1, line: 1, baseType: !10, size: 32, align: 32, offset: 32)
!17 = !DIExpression()
!18 = !DILocation(line: 11, column: 18, scope: !7)
!19 = !DIExpression(DW_OP_LLVM_fragment, 0, 32)
!20 = !DIExpression(DW_OP_LLVM_fragment, 32, 32)
!21 = !DILocation(line: 12, column: 23, scope: !22)
!22 = distinct !DILexicalBlock(scope: !23, file: !1, line: 12, column: 3)
!23 = distinct !DILexicalBlock(scope: !7, file: !1, line: 12, column: 3)
!24 = !{!25, !25, i64 0}
!25 = !{!"int", !26, i64 0}
!26 = !{!"omnipotent char", !27, i64 0}
!27 = !{!"Simple C/C++ TBAA"}
!28 = !DILocation(line: 12, column: 19, scope: !22)
!29 = !DILocation(line: 12, column: 17, scope: !22)
!30 = !DILocation(line: 12, column: 3, scope: !23)
!31 = !DILocation(line: 13, column: 11, scope: !32)
!32 = distinct !DILexicalBlock(scope: !22, file: !1, line: 12, column: 27)
!33 = !DILocation(line: 14, column: 11, scope: !32)
!34 = distinct !{!34, !35}
!35 = !DILocation(line: 12, column: 3, scope: !7)
!36 = !DILocation(line: 16, column: 14, scope: !7)
!37 = !DILocation(line: 16, column: 3, scope: !7)
!38 = distinct !DISubprogram(name: "pad_right", scope: !1, file: !1, line: 19, type: !39, isLocal: false, isDefinition: true, scopeLine: 19, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !46)
!39 = !DISubroutineType(types: !40)
!40 = !{!10, !41}
!41 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "PadRight", file: !1, line: 2, size: 64, align: 32, elements: !42)
!42 = !{!43, !45}
!43 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !41, file: !1, line: 2, baseType: !44, size: 32, align: 32)
!44 = !DIBasicType(name: "long int", size: 32, align: 32, encoding: DW_ATE_signed)
!45 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !41, file: !1, line: 2, baseType: !10, size: 32, align: 32, offset: 32)
!46 = !{!47}
!47 = !DILocalVariable(name: "o", arg: 1, scope: !38, file: !1, line: 19, type: !41)
!48 = !DILocation(line: 19, column: 31, scope: !38)
!49 = !DILocation(line: 20, column: 3, scope: !38)
!50 = distinct !DISubprogram(name: "pad_left", scope: !1, file: !1, line: 23, type: !51, isLocal: false, isDefinition: true, scopeLine: 23, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !57)
!51 = !DISubroutineType(types: !52)
!52 = !{!10, !53}
!53 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "PadLeft", file: !1, line: 3, size: 64, align: 32, elements: !54)
!54 = !{!55, !56}
!55 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !53, file: !1, line: 3, baseType: !10, size: 32, align: 32)
!56 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !53, file: !1, line: 3, baseType: !44, size: 32, align: 32, offset: 32)
!57 = !{!58}
!58 = !DILocalVariable(name: "o", arg: 1, scope: !50, file: !1, line: 23, type: !53)
!59 = !DILocation(line: 23, column: 29, scope: !50)
!60 = !DILocation(line: 24, column: 3, scope: !50)
!61 = distinct !DISubprogram(name: "nested", scope: !1, file: !1, line: 27, type: !62, isLocal: false, isDefinition: true, scopeLine: 27, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !70)
!62 = !DISubroutineType(types: !63)
!63 = !{!10, !64}
!64 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Nested", file: !1, line: 4, size: 128, align: 32, elements: !65)
!65 = !{!66}
!66 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !64, file: !1, line: 4, baseType: !67, size: 128, align: 32)
!67 = !DICompositeType(tag: DW_TAG_array_type, baseType: !53, size: 128, align: 32, elements: !68)
!68 = !{!69}
!69 = !DISubrange(count: 2)
!70 = !{!71, !72}
!71 = !DILocalVariable(name: "o", arg: 1, scope: !61, file: !1, line: 27, type: !64)
!72 = !DILocalVariable(name: "p", scope: !61, file: !1, line: 28, type: !53)
!73 = !DIExpression(DW_OP_deref)
!74 = !DILocation(line: 27, column: 26, scope: !61)
!75 = !DILocation(line: 28, column: 18, scope: !61)
!76 = !DILocation(line: 28, column: 22, scope: !61)
!77 = !DILocation(line: 29, column: 3, scope: !61)
!78 = distinct !DISubprogram(name: "bitpiece_spill", scope: !1, file: !1, line: 32, type: !8, isLocal: false, isDefinition: true, scopeLine: 32, isOptimized: true, unit: !0, retainedNodes: !79)
!79 = !{!80}
!80 = !DILocalVariable(name: "o", scope: !78, file: !1, line: 33, type: !13)
!81 = !DILocation(line: 33, column: 18, scope: !78)
!82 = !DILocation(line: 33, column: 26, scope: !78)
!83 = !DILocation(line: 35, column: 3, scope: !78)
!84 = !{i32 603}
!85 = !DILocation(line: 37, column: 3, scope: !78)
