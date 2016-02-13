; RUN: llc < %s | FileCheck %s --check-prefix=ASM
; RUN: llc < %s -filetype=obj | llvm-readobj -codeview - | FileCheck %s --check-prefix=OBJ

; Generated from:
; volatile int x;
; int getint(void);
; void putint(int);
; static inline int inlineinc(int a) {
;   int b = a + 1;
;   ++x;
;   return b;
; }
; void f(int p) {
;   if (p) {
;     int a = getint();
;     int b = inlineinc(a);
;     putint(b);
;   } else {
;     int c = getint();
;     putint(c);
;   }
; }

; ASM: f:                                      # @f
; ASM: .Lfunc_begin0:
; ASM: # BB#0:                                 # %entry
; ASM:         pushq   %rsi
; ASM:         subq    $32, %rsp
; ASM:         #DEBUG_VALUE: f:p <- %ECX
; ASM:         movl    %ecx, %esi
; ASM: [[p_ecx_esi:\.Ltmp.*]]:
; ASM:         #DEBUG_VALUE: f:p <- %ESI
; ASM:         callq   getint
; ASM: [[after_getint:\.Ltmp.*]]:
; ASM:         #DEBUG_VALUE: a <- %EAX
; ASM:         #DEBUG_VALUE: inlineinc:a <- %EAX
; ASM:         #DEBUG_VALUE: c <- %EAX
; ASM:         testl   %esi, %esi
; ASM:         je      .LBB0_2
; ASM: # BB#1:                                 # %if.then
; ASM:         #DEBUG_VALUE: c <- %EAX
; ASM:         #DEBUG_VALUE: inlineinc:a <- %EAX
; ASM:         #DEBUG_VALUE: a <- %EAX
; ASM:         #DEBUG_VALUE: f:p <- %ESI
; ASM:         incl    %eax
; ASM: [[after_inc_eax:\.Ltmp.*]]:
; ASM:         #DEBUG_VALUE: inlineinc:b <- %EAX
; ASM:         #DEBUG_VALUE: b <- %EAX
; ASM:         incl    x(%rip)
; ASM: [[after_if:\.Ltmp.*]]:
; ASM: .LBB0_2:                                # %if.else
; ASM:         #DEBUG_VALUE: f:p <- %ESI
; ASM:         movl    %eax, %ecx
; ASM:         addq    $32, %rsp
; ASM:         popq    %rsi
; ASM: [[func_end:\.Ltmp.*]]:
; ASM:         rex64 jmp       putint          # TAILCALL

; ASM:         .short  4414                    # Record kind: S_LOCAL
; ASM:         .asciz  "p"
; ASM:         .cv_def_range    .Lfunc_begin0 [[p_ecx_esi]], "A\021\022\000\000\000"
; ASM:         .cv_def_range    [[p_ecx_esi]] [[func_end]], "A\021\027\000\000\000"
; ASM:         .short  4414                    # Record kind: S_LOCAL
; ASM:         .asciz  "a"
; ASM:         .cv_def_range    [[after_getint]] [[after_inc_eax]], "A\021\021\000\000\000"
; ASM:         .short  4414                    # Record kind: S_LOCAL
; ASM:         .asciz  "c"
; ASM:         .cv_def_range    [[after_getint]] [[after_inc_eax]], "A\021\021\000\000\000"
; ASM:         .short  4414                    # Record kind: S_LOCAL
; ASM:         .asciz  "b"
; ASM:         .cv_def_range    [[after_inc_eax]] [[after_if]], "A\021\021\000\000\000"

; ASM:         .short  4429                    # Record kind: S_INLINESITE
; ASM:         .short  4414                    # Record kind: S_LOCAL
; ASM:         .asciz  "a"
; ASM:         .cv_def_range    [[after_getint]] [[after_inc_eax]], "A\021\021\000\000\000"
; ASM:         .short  4414                    # Record kind: S_LOCAL
; ASM:         .asciz  "b"
; ASM:         .cv_def_range    [[after_inc_eax]] [[after_if]], "A\021\021\000\000\000"
; ASM:         .short  4430                    # Record kind: S_INLINESITE_END

; OBJ: Subsection [
; OBJ:   SubSectionType: Symbols (0xF1)
; OBJ:   ProcStart {
; OBJ:     DisplayName: f
; OBJ:   }
; OBJ:   Local {
; OBJ:     Type: int (0x74)
; OBJ:     Flags [ (0x1)
; OBJ:       IsParameter (0x1)
; OBJ:     ]
; OBJ:     VarName: p
; OBJ:   }
; OBJ:   DefRangeRegister {
; OBJ:     Register: 18
; OBJ:     LocalVariableAddrRange {
; OBJ:       OffsetStart: .text+0x0
; OBJ:       ISectStart: 0x0
; OBJ:       Range: 0x7
; OBJ:     }
; OBJ:   }
; OBJ:   DefRangeRegister {
; OBJ:     Register: 23
; OBJ:     LocalVariableAddrRange {
; OBJ:       OffsetStart: .text+0x7
; OBJ:       ISectStart: 0x0
; OBJ:       Range: 0x18
; OBJ:     }
; OBJ:   }
; OBJ:   Local {
; OBJ:     Type: int (0x74)
; OBJ:     Flags [ (0x0)
; OBJ:     ]
; OBJ:     VarName: a
; OBJ:   }
; OBJ:   DefRangeRegister {
; OBJ:     Register: 17
; OBJ:     LocalVariableAddrRange {
; OBJ:       OffsetStart: .text+0xC
; OBJ:       ISectStart: 0x0
; OBJ:       Range: 0x6
; OBJ:     }
; OBJ:   }
; OBJ:   Local {
; OBJ:     Type: int (0x74)
; OBJ:     Flags [ (0x0)
; OBJ:     ]
; OBJ:     VarName: c
; OBJ:   }
; OBJ:   DefRangeRegister {
; OBJ:     Register: 17
; OBJ:     LocalVariableAddrRange {
; OBJ:       OffsetStart: .text+0xC
; OBJ:       ISectStart: 0x0
; OBJ:       Range: 0x6
; OBJ:     }
; OBJ:   }
; OBJ:   Local {
; OBJ:     Type: int (0x74)
; OBJ:     Flags [ (0x0)
; OBJ:     ]
; OBJ:     VarName: b
; OBJ:   }
; OBJ:   DefRangeRegister {
; OBJ:     Register: 17
; OBJ:     LocalVariableAddrRange {
; OBJ:       OffsetStart: .text+0x12
; OBJ:       ISectStart: 0x0
; OBJ:       Range: 0x6
; OBJ:     }
; OBJ:   }
; OBJ:   InlineSite {
; OBJ:     PtrParent: 0x0
; OBJ:     PtrEnd: 0x0
; OBJ:     Inlinee: inlineinc (0x1003)
; OBJ:   }
; OBJ:   Local {
; OBJ:     Type: int (0x74)
; OBJ:     Flags [ (0x1)
; OBJ:       IsParameter (0x1)
; OBJ:     ]
; OBJ:     VarName: a
; OBJ:   }
; OBJ:   DefRangeRegister {
; OBJ:     Register: 17
; OBJ:     LocalVariableAddrRange {
; OBJ:       OffsetStart: .text+0xC
; OBJ:       ISectStart: 0x0
; OBJ:       Range: 0x6
; OBJ:     }
; OBJ:   }
; OBJ:   Local {
; OBJ:     Type: int (0x74)
; OBJ:     Flags [ (0x0)
; OBJ:     ]
; OBJ:     VarName: b
; OBJ:   }
; OBJ:   DefRangeRegister {
; OBJ:     Register: 17
; OBJ:     LocalVariableAddrRange {
; OBJ:       OffsetStart: .text+0x12
; OBJ:       ISectStart: 0x0
; OBJ:       Range: 0x6
; OBJ:     }
; OBJ:   }
; OBJ:   InlineSiteEnd {
; OBJ:   }
; OBJ:   ProcEnd
; OBJ: ]

; ModuleID = 't.cpp'
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc18.0.0"

@x = internal global i32 0, align 4

; Function Attrs: nounwind uwtable
define void @f(i32 %p) #0 !dbg !4 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %p, i64 0, metadata !9, metadata !29), !dbg !30
  %tobool = icmp eq i32 %p, 0, !dbg !31
  %call2 = tail call i32 @getint() #3, !dbg !32
  br i1 %tobool, label %if.else, label %if.then, !dbg !33

if.then:                                          ; preds = %entry
  tail call void @llvm.dbg.value(metadata i32 %call2, i64 0, metadata !10, metadata !29), !dbg !34
  tail call void @llvm.dbg.value(metadata i32 %call2, i64 0, metadata !20, metadata !29), !dbg !35
  %add.i = add nsw i32 %call2, 1, !dbg !37
  tail call void @llvm.dbg.value(metadata i32 %add.i, i64 0, metadata !21, metadata !29), !dbg !38
  %0 = load volatile i32, i32* @x, align 4, !dbg !39, !tbaa !40
  %inc.i = add nsw i32 %0, 1, !dbg !39
  store volatile i32 %inc.i, i32* @x, align 4, !dbg !39, !tbaa !40
  tail call void @llvm.dbg.value(metadata i32 %add.i, i64 0, metadata !13, metadata !29), !dbg !44
  tail call void @putint(i32 %add.i) #3, !dbg !45
  br label %if.end, !dbg !46

if.else:                                          ; preds = %entry
  tail call void @llvm.dbg.value(metadata i32 %call2, i64 0, metadata !14, metadata !29), !dbg !47
  tail call void @putint(i32 %call2) #3, !dbg !48
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void, !dbg !49
}

declare i32 @getint() #1

declare void @putint(i32) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!25, !26, !27}
!llvm.ident = !{!28}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 (trunk 260617) (llvm/trunk 260619)", isOptimized: true, runtimeVersion: 0, emissionKind: 1, enums: !2, subprograms: !3, globals: !22)
!1 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!2 = !{}
!3 = !{!4, !16}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 9, type: !5, isLocal: false, isDefinition: true, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: true, variables: !8)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{!9, !10, !13, !14}
!9 = !DILocalVariable(name: "p", arg: 1, scope: !4, file: !1, line: 9, type: !7)
!10 = !DILocalVariable(name: "a", scope: !11, file: !1, line: 11, type: !7)
!11 = distinct !DILexicalBlock(scope: !12, file: !1, line: 10, column: 10)
!12 = distinct !DILexicalBlock(scope: !4, file: !1, line: 10, column: 7)
!13 = !DILocalVariable(name: "b", scope: !11, file: !1, line: 12, type: !7)
!14 = !DILocalVariable(name: "c", scope: !15, file: !1, line: 15, type: !7)
!15 = distinct !DILexicalBlock(scope: !12, file: !1, line: 14, column: 10)
!16 = distinct !DISubprogram(name: "inlineinc", scope: !1, file: !1, line: 4, type: !17, isLocal: true, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: true, variables: !19)
!17 = !DISubroutineType(types: !18)
!18 = !{!7, !7}
!19 = !{!20, !21}
!20 = !DILocalVariable(name: "a", arg: 1, scope: !16, file: !1, line: 4, type: !7)
!21 = !DILocalVariable(name: "b", scope: !16, file: !1, line: 5, type: !7)
!22 = !{!23}
!23 = !DIGlobalVariable(name: "x", scope: !0, file: !1, line: 1, type: !24, isLocal: false, isDefinition: true, variable: i32* @x)
!24 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !7)
!25 = !{i32 2, !"CodeView", i32 1}
!26 = !{i32 2, !"Debug Info Version", i32 3}
!27 = !{i32 1, !"PIC Level", i32 2}
!28 = !{!"clang version 3.9.0 (trunk 260617) (llvm/trunk 260619)"}
!29 = !DIExpression()
!30 = !DILocation(line: 9, column: 12, scope: !4)
!31 = !DILocation(line: 10, column: 7, scope: !12)
!32 = !DILocation(line: 15, column: 13, scope: !15)
!33 = !DILocation(line: 10, column: 7, scope: !4)
!34 = !DILocation(line: 11, column: 9, scope: !11)
!35 = !DILocation(line: 4, column: 33, scope: !16, inlinedAt: !36)
!36 = distinct !DILocation(line: 12, column: 13, scope: !11)
!37 = !DILocation(line: 5, column: 13, scope: !16, inlinedAt: !36)
!38 = !DILocation(line: 5, column: 7, scope: !16, inlinedAt: !36)
!39 = !DILocation(line: 6, column: 3, scope: !16, inlinedAt: !36)
!40 = !{!41, !41, i64 0}
!41 = !{!"int", !42, i64 0}
!42 = !{!"omnipotent char", !43, i64 0}
!43 = !{!"Simple C/C++ TBAA"}
!44 = !DILocation(line: 12, column: 9, scope: !11)
!45 = !DILocation(line: 13, column: 5, scope: !11)
!46 = !DILocation(line: 14, column: 3, scope: !11)
!47 = !DILocation(line: 15, column: 9, scope: !15)
!48 = !DILocation(line: 16, column: 5, scope: !15)
!49 = !DILocation(line: 18, column: 1, scope: !4)
