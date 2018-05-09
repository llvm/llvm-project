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
; ASM: # %bb.0:                                 # %entry
; ASM:         pushq   %rsi
; ASM:         subq    $32, %rsp
; ASM:         #DEBUG_VALUE: f:p <- $ecx
; ASM:         movl    %ecx, %esi
; ASM: [[p_ecx_esi:\.Ltmp.*]]:
; ASM:         #DEBUG_VALUE: f:p <- $esi
; ASM:         callq   getint
; ASM: [[after_getint:\.Ltmp.*]]:
; ASM:         #DEBUG_VALUE: a <- $eax
; ASM:         #DEBUG_VALUE: inlineinc:a <- $eax
; ASM:         #DEBUG_VALUE: c <- $eax
; ASM:         testl   %esi, %esi
; ASM:         je      .LBB0_2
; ASM: [[after_je:\.Ltmp.*]]:
; ASM: # %bb.1:                                 # %if.then
; ASM-DAG:     #DEBUG_VALUE: inlineinc:a <- $eax
; ASM-DAG:     #DEBUG_VALUE: a <- $eax
; ASM-DAG:     #DEBUG_VALUE: f:p <- $esi
; ASM:         addl    $1, %eax
; ASM: [[after_inc_eax:\.Ltmp.*]]:
; ASM:         #DEBUG_VALUE: inlineinc:b <- $eax
; ASM:         #DEBUG_VALUE: b <- $eax
; ASM:         addl    $1, x(%rip)
; ASM: [[after_if:\.Ltmp.*]]:
; ASM: .LBB0_2:                                # %if.else
; ASM:         #DEBUG_VALUE: f:p <- $esi
; ASM:         movl    %eax, %ecx
; ASM:         addq    $32, %rsp
; ASM:         popq    %rsi
; ASM: [[func_end:\.Ltmp.*]]:
; ASM:         jmp     putint                  # TAILCALL

; ASM:         .short  4414                    # Record kind: S_LOCAL
; ASM:         .asciz  "p"
; ASM:         .cv_def_range    .Lfunc_begin0 [[p_ecx_esi]], "A\021\022\000\000\000"
; ASM:         .cv_def_range    [[p_ecx_esi]] [[func_end]], "A\021\027\000\000\000"
; ASM:         .short  4414                    # Record kind: S_LOCAL
; ASM:         .asciz  "c"
; ASM:         .cv_def_range    [[after_getint]] [[after_je]], "A\021\021\000\000\000"
; ASM:         .short  4414                    # Record kind: S_LOCAL
; ASM:         .asciz  "a"
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
; OBJ:   {{.*}}Proc{{.*}}Sym {
; OBJ:     DisplayName: f
; OBJ:   }
; OBJ:   LocalSym {
; OBJ:     Type: int (0x74)
; OBJ:     Flags [ (0x1)
; OBJ:       IsParameter (0x1)
; OBJ:     ]
; OBJ:     VarName: p
; OBJ:   }
; OBJ:   DefRangeRegisterSym {
; OBJ:     Register: ECX (0x12)
; OBJ:     LocalVariableAddrRange {
; OBJ:       OffsetStart: .text+0x0
; OBJ:       ISectStart: 0x0
; OBJ:       Range: 0x7
; OBJ:     }
; OBJ:   }
; OBJ:   DefRangeRegisterSym {
; OBJ:     Register: ESI (0x17)
; OBJ:     LocalVariableAddrRange {
; OBJ:       OffsetStart: .text+0x7
; OBJ:       ISectStart: 0x0
; OBJ:       Range: 0x1A
; OBJ:     }
; OBJ:   }
; OBJ:   LocalSym {
; OBJ:     Type: int (0x74)
; OBJ:     Flags [ (0x0)
; OBJ:     ]
; OBJ:     VarName: c
; OBJ:   }
; OBJ:   DefRangeRegisterSym {
; OBJ:     Register: EAX (0x11)
; OBJ:     LocalVariableAddrRange {
; OBJ:       OffsetStart: .text+0xC
; OBJ:       ISectStart: 0x0
; OBJ:       Range: 0x4
; OBJ:     }
; OBJ:   }
; OBJ:   LocalSym {
; OBJ:     Type: int (0x74)
; OBJ:     Flags [ (0x0)
; OBJ:     ]
; OBJ:     VarName: a
; OBJ:   }
; OBJ:   DefRangeRegisterSym {
; OBJ:     Register: EAX (0x11)
; OBJ:     LocalVariableAddrRange {
; OBJ:       OffsetStart: .text+0xC
; OBJ:       ISectStart: 0x0
; OBJ:       Range: 0x7
; OBJ:     }
; OBJ:   }
; OBJ:   LocalSym {
; OBJ:     Type: int (0x74)
; OBJ:     Flags [ (0x0)
; OBJ:     ]
; OBJ:     VarName: b
; OBJ:   }
; OBJ:   DefRangeRegisterSym {
; OBJ:     Register: EAX (0x11)
; OBJ:     MayHaveNoName: 0
; OBJ:       OffsetStart: .text+0x13
; OBJ:       ISectStart: 0x0
; OBJ:       Range: 0x7
; OBJ:     }
; OBJ:   }
; OBJ:   InlineSiteSym {
; OBJ:     PtrParent: 0x0
; OBJ:     PtrEnd: 0x0
; OBJ:     Inlinee: inlineinc (0x1002)
; OBJ:   }
; OBJ:   LocalSym {
; OBJ:     Type: int (0x74)
; OBJ:     Flags [ (0x1)
; OBJ:       IsParameter (0x1)
; OBJ:     ]
; OBJ:     VarName: a
; OBJ:   }
; OBJ:   DefRangeRegisterSym {
; OBJ:     Register: EAX (0x11)
; OBJ:     LocalVariableAddrRange {
; OBJ:       OffsetStart: .text+0xC
; OBJ:       ISectStart: 0x0
; OBJ:       Range: 0x7
; OBJ:     }
; OBJ:   }
; OBJ:   LocalSym {
; OBJ:     Type: int (0x74)
; OBJ:     Flags [ (0x0)
; OBJ:     ]
; OBJ:     VarName: b
; OBJ:   }
; OBJ:   DefRangeRegisterSym {
; OBJ:     Register: EAX (0x11)
; OBJ:     LocalVariableAddrRange {
; OBJ:       OffsetStart: .text+0x13
; OBJ:       ISectStart: 0x0
; OBJ:       Range: 0x7
; OBJ:     }
; OBJ:   }
; OBJ:   InlineSiteEnd {
; OBJ:   }
; OBJ:   ProcEnd
; OBJ: ]

; ModuleID = 't.cpp'
source_filename = "test/DebugInfo/COFF/register-variables.ll"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc18.0.0"

@x = internal global i32 0, align 4, !dbg !0

; Function Attrs: nounwind uwtable
define void @f(i32 %p) #0 !dbg !12 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %p, metadata !16, metadata !23), !dbg !24
  %tobool = icmp eq i32 %p, 0, !dbg !25
  %call2 = tail call i32 @getint() #3, !dbg !26
  br i1 %tobool, label %if.else, label %if.then, !dbg !27

if.then:                                          ; preds = %entry
  tail call void @llvm.dbg.value(metadata i32 %call2, metadata !17, metadata !23), !dbg !28
  tail call void @llvm.dbg.value(metadata i32 %call2, metadata !29, metadata !23), !dbg !35
  %add.i = add nsw i32 %call2, 1, !dbg !37
  tail call void @llvm.dbg.value(metadata i32 %add.i, metadata !34, metadata !23), !dbg !38
  %0 = load volatile i32, i32* @x, align 4, !dbg !39, !tbaa !40
  %inc.i = add nsw i32 %0, 1, !dbg !39
  store volatile i32 %inc.i, i32* @x, align 4, !dbg !39, !tbaa !40
  tail call void @llvm.dbg.value(metadata i32 %add.i, metadata !20, metadata !23), !dbg !44
  tail call void @putint(i32 %add.i) #3, !dbg !45
  br label %if.end, !dbg !46

if.else:                                          ; preds = %entry
  tail call void @llvm.dbg.value(metadata i32 %call2, metadata !21, metadata !23), !dbg !47
  tail call void @putint(i32 %call2) #3, !dbg !48
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void, !dbg !49
}

declare i32 @getint() #1

declare void @putint(i32) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9, !10}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 3.9.0 (trunk 260617) (llvm/trunk 260619)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !7)
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"CodeView", i32 1}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"PIC Level", i32 2}
!11 = !{!"clang version 3.9.0 (trunk 260617) (llvm/trunk 260619)"}
!12 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 9, type: !13, isLocal: false, isDefinition: true, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !15)
!13 = !DISubroutineType(types: !14)
!14 = !{null, !7}
!15 = !{!16, !17, !20, !21}
!16 = !DILocalVariable(name: "p", arg: 1, scope: !12, file: !3, line: 9, type: !7)
!17 = !DILocalVariable(name: "a", scope: !18, file: !3, line: 11, type: !7)
!18 = distinct !DILexicalBlock(scope: !19, file: !3, line: 10, column: 10)
!19 = distinct !DILexicalBlock(scope: !12, file: !3, line: 10, column: 7)
!20 = !DILocalVariable(name: "b", scope: !18, file: !3, line: 12, type: !7)
!21 = !DILocalVariable(name: "c", scope: !22, file: !3, line: 15, type: !7)
!22 = distinct !DILexicalBlock(scope: !19, file: !3, line: 14, column: 10)
!23 = !DIExpression()
!24 = !DILocation(line: 9, column: 12, scope: !12)
!25 = !DILocation(line: 10, column: 7, scope: !19)
!26 = !DILocation(line: 15, column: 13, scope: !22)
!27 = !DILocation(line: 10, column: 7, scope: !12)
!28 = !DILocation(line: 11, column: 9, scope: !18)
!29 = !DILocalVariable(name: "a", arg: 1, scope: !30, file: !3, line: 4, type: !7)
!30 = distinct !DISubprogram(name: "inlineinc", scope: !3, file: !3, line: 4, type: !31, isLocal: true, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !33)
!31 = !DISubroutineType(types: !32)
!32 = !{!7, !7}
!33 = !{!29, !34}
!34 = !DILocalVariable(name: "b", scope: !30, file: !3, line: 5, type: !7)
!35 = !DILocation(line: 4, column: 33, scope: !30, inlinedAt: !36)
!36 = distinct !DILocation(line: 12, column: 13, scope: !18)
!37 = !DILocation(line: 5, column: 13, scope: !30, inlinedAt: !36)
!38 = !DILocation(line: 5, column: 7, scope: !30, inlinedAt: !36)
!39 = !DILocation(line: 6, column: 3, scope: !30, inlinedAt: !36)
!40 = !{!41, !41, i64 0}
!41 = !{!"int", !42, i64 0}
!42 = !{!"omnipotent char", !43, i64 0}
!43 = !{!"Simple C/C++ TBAA"}
!44 = !DILocation(line: 12, column: 9, scope: !18)
!45 = !DILocation(line: 13, column: 5, scope: !18)
!46 = !DILocation(line: 14, column: 3, scope: !18)
!47 = !DILocation(line: 15, column: 9, scope: !22)
!48 = !DILocation(line: 16, column: 5, scope: !22)
!49 = !DILocation(line: 18, column: 1, scope: !12)

