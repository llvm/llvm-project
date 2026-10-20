; RUN: llc < %s -filetype=obj | llvm-readobj --codeview - | FileCheck %s

; The point of this is mostly just to avoid crashing... there isn't any way
; to encode most of the information we want to encode.  But we try to do what
; we can.
;
; Generated from:
;
; #include <arm_sve.h>
; void g();
; svint32_t f(svint32_t aaa, svint32_t bbb, svint32_t *ccc) {
;   asm("":::"z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9",
;            "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18",
;            "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27",
;            "z28", "z29", "z30", "z31");
;  return aaa**ccc+bbb;
;}

; Emit the SVE type.  We represent this as an array with unknown bound.

; CHECK:      Array (0x1000) {
; CHECK-NEXT:     TypeLeafKind: LF_ARRAY (0x1503)
; CHECK-NEXT:     ElementType: int (0x74)
; CHECK-NEXT:     IndexType: unsigned __int64 (0x23)
; CHECK-NEXT:     SizeOf: 0
; CHECK-NEXT:     Name:
; CHECK-NEXT:   }

; Emit frame information.  This is missing the size of the SVE
; variables, but we can't really do anything about that.

; CHECK:         FrameProcSym {
; CHECK-NEXT:      Kind: S_FRAMEPROC (0x1012)
; CHECK-NEXT:      TotalFrameBytes: 0x10
; CHECK-NEXT:      PaddingFrameBytes: 0x0
; CHECK-NEXT:      OffsetToPadding: 0x0
; CHECK-NEXT:      BytesOfCalleeSavedRegisters: 0x0
; CHECK-NEXT:      OffsetOfExceptionHandler: 0x0
; CHECK-NEXT:      SectionIdOfExceptionHandler: 0x0
; CHECK-NEXT:      Flags [ (0x116008)
; CHECK-NEXT:        HasInlineAssembly (0x8)
; CHECK-NEXT:        OptimizedForSpeed (0x100000)
; CHECK-NEXT:        SafeBuffers (0x2000)
; CHECK-NEXT:      ]
; CHECK-NEXT:      LocalFramePtrReg: ARM64_NOREG (0x0)
; CHECK-NEXT:      ParamFramePtrReg: ARM64_NOREG (0x0)
; CHECK-NEXT:    }

; Emit the symbols for the local variables.
;
; ccc is a normal pointer.
;
; We can't represent bbb anywhere in its range; there's no way to name Z
; registers, and no way to express its location on the stack relative
; to the stack pointer when it's spilled.
;
; In the middle of the range, aaa happens to have a scalable offset of zero,
; so we can represent it while it's on the stack.

; CHECK-NEXT:    LocalSym {
; CHECK-NEXT:      Kind: S_LOCAL (0x113E)
; CHECK-NEXT:      Type: 0x1000
; CHECK-NEXT:      Flags [ (0x1)
; CHECK-NEXT:        IsParameter (0x1)
; CHECK-NEXT:      ]
; CHECK-NEXT:      VarName: aaa
; CHECK-NEXT:    }
; CHECK-NEXT:    DefRangeRegisterRelSym {
; CHECK-NEXT:      Kind: S_DEFRANGE_REGISTER_REL (0x1145)
; CHECK-NEXT:      BaseRegister: ARM64_SP (0x51)
; CHECK-NEXT:      HasSpilledUDTMember: No
; CHECK-NEXT:      OffsetInParent: 0
; CHECK-NEXT:      BasePointerOffset: 0
; CHECK-NEXT:      LocalVariableAddrRange {
; CHECK-NEXT:        OffsetStart: .text+0x58
; CHECK-NEXT:        ISectStart: 0x0
; CHECK-NEXT:        Range: 0xC
; CHECK-NEXT:      }
; CHECK-NEXT:    }
; CHECK-NEXT:    LocalSym {
; CHECK-NEXT:      Kind: S_LOCAL (0x113E)
; CHECK-NEXT:      Type: 0x1000
; CHECK-NEXT:      Flags [ (0x101)
; CHECK-NEXT:        IsOptimizedOut (0x100)
; CHECK-NEXT:        IsParameter (0x1)
; CHECK-NEXT:      ]
; CHECK-NEXT:      VarName: bbb
; CHECK-NEXT:    }
; CHECK-NEXT:    LocalSym {
; CHECK-NEXT:      Kind: S_LOCAL (0x113E)
; CHECK-NEXT:      Type: * (0x1001)
; CHECK-NEXT:      Flags [ (0x1)
; CHECK-NEXT:        IsParameter (0x1)
; CHECK-NEXT:      ]
; CHECK-NEXT:      VarName: ccc
; CHECK-NEXT:    }
; CHECK-NEXT:    DefRangeRegisterSym {
; CHECK-NEXT:      Kind: S_DEFRANGE_REGISTER (0x1141)
; CHECK-NEXT:      Register: ARM64_X0 (0x32)
; CHECK-NEXT:      MayHaveNoName: 0
; CHECK-NEXT:      LocalVariableAddrRange {
; CHECK-NEXT:        OffsetStart: .text+0x0
; CHECK-NEXT:        ISectStart: 0x0
; CHECK-NEXT:        Range: 0xB8
; CHECK-NEXT:      }
; CHECK-NEXT:    }
; CHECK-NEXT:    ProcEnd {
; CHECK-NEXT:      Kind: S_PROC_ID_END (0x114F)
; CHECK-NEXT:    }

target triple = "aarch64-unknown-windows-msvc19.33.0"

; Function Attrs: mustprogress nounwind uwtable vscale_range(1,16)
define dso_local <vscale x 4 x i32> @"?f@@YAU__SVInt32_t@__clang@@U12@0PEAU12@@Z"(<vscale x 4 x i32> %aaa, <vscale x 4 x i32> %bbb, ptr noundef readonly captures(none) %ccc) local_unnamed_addr #0 !dbg !10 {
entry:
    #dbg_value(ptr %ccc, !23, !DIExpression(), !26)
    #dbg_value(<vscale x 4 x i32> %bbb, !24, !DIExpression(), !26)
    #dbg_value(<vscale x 4 x i32> %aaa, !25, !DIExpression(), !26)
  tail call void asm sideeffect "", "~{z0},~{z1},~{z2},~{z3},~{z4},~{z5},~{z6},~{z7},~{z8},~{z9},~{z10},~{z11},~{z12},~{z13},~{z14},~{z15},~{z16},~{z17},~{z18},~{z19},~{z20},~{z21},~{z22},~{z23},~{z24},~{z25},~{z26},~{z27},~{z28},~{z29},~{z30},~{z31}"() #1, !dbg !27, !srcloc !28
  %0 = load <vscale x 4 x i32>, ptr %ccc, align 16, !dbg !27
  %mul = mul <vscale x 4 x i32> %0, %aaa, !dbg !27
  %add = add <vscale x 4 x i32> %mul, %bbb, !dbg !27
  ret <vscale x 4 x i32> %add, !dbg !27
}

attributes #0 = { mustprogress nounwind uwtable vscale_range(1,16) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+fullfp16,+neon,+sve,+v8a,-fmv" }
attributes #1 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 21.0.0git", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "-", directory: "", checksumkind: CSK_MD5, checksum: "e54fc2ba768e4a43f64b8a9d03a374d6")
!2 = !{i32 2, !"CodeView", i32 1}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 2}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"uwtable", i32 2}
!7 = !{i32 7, !"frame-pointer", i32 1}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang version 21.0.0git"}
!10 = distinct !DISubprogram(name: "f", linkageName: "?f@@YAU__SVInt32_t@__clang@@U12@0PEAU12@@Z", scope: !11, file: !11, line: 2, type: !12, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !22)
!11 = !DIFile(filename: "<stdin>", directory: "", checksumkind: CSK_MD5, checksum: "e54fc2ba768e4a43f64b8a9d03a374d6")
!12 = !DISubroutineType(types: !13)
!13 = !{!14, !14, !14, !21}
!14 = !DIDerivedType(tag: DW_TAG_typedef, name: "svint32_t", file: !15, line: 30, baseType: !16)
!15 = !DIFile(filename: "arm_sve.h", directory: "", checksumkind: CSK_MD5, checksum: "34027e9d24f4b03c6e5370869d5cc907")
!16 = !DIDerivedType(tag: DW_TAG_typedef, name: "__SVInt32_t", file: !1, baseType: !17)
!17 = !DICompositeType(tag: DW_TAG_array_type, baseType: !18, flags: DIFlagVector, elements: !19)
!18 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!19 = !{!20}
!20 = !DISubrange(lowerBound: 0, upperBound: !DIExpression(DW_OP_constu, 2, DW_OP_bregx, 46, 0, DW_OP_mul, DW_OP_constu, 1, DW_OP_minus))
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!22 = !{!23, !24, !25}
!23 = !DILocalVariable(name: "ccc", arg: 3, scope: !10, file: !11, line: 2, type: !21)
!24 = !DILocalVariable(name: "bbb", arg: 2, scope: !10, file: !11, line: 2, type: !14)
!25 = !DILocalVariable(name: "aaa", arg: 1, scope: !10, file: !11, line: 2, type: !14)
!26 = !DILocation(line: 0, scope: !10)
!27 = !DILocation(line: 2, scope: !10)
!28 = !{i64 98}
