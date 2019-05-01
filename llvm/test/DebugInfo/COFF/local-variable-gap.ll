; RUN: llc -mtriple=x86_64-windows-msvc < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=x86_64-windows-msvc < %s -filetype=obj | llvm-readobj --codeview - | FileCheck %s --check-prefix=OBJ

; This test attempts to exercise gaps in local variables. The local variable 'p'
; will end up in some CSR (esi), which will be used in both the BB scheduled
; discontiguously out of line and the normal return BB. The best way to encode
; this is to use a LocalVariableAddrGap. If the gap is too large, multiple
; ranges should be emitted.

; Source to regenerate:
; int barrier();
; int vardef();
; void use(int);
; void __declspec(noreturn) call_noreturn(int);
; int f() {
;   if (barrier()) {
;     int p = vardef();
;     if (barrier()) // Unlikely, will be placed after return
;       call_noreturn(p);
;     use(p);
;   } else {
;     barrier();
;   }
;   return 0;
; }

; ASM: f:                                      # @f
; ASM:         pushq   %rsi
; ASM:         subq    $32, %rsp
; ASM:         callq   barrier
; ASM:         testl   %eax, %eax
; ASM:         je      .LBB0_3
; ASM:         callq   vardef
; ASM:         movl    %eax, %esi
; ASM: [[p_b1:\.Ltmp[0-9]+]]:
; ASM:         #DEBUG_VALUE: p <- $esi
; ASM:         callq   barrier
; ASM:         movl    %esi, %ecx
; ASM:         testl   %eax, %eax
; ASM:         jne     .LBB0_5
; ASM: # %bb.2:                                 # %if.end
; ASM:         #DEBUG_VALUE: p <- $esi
; ASM:         callq   use
; ASM:         jmp     .LBB0_4
; ASM: [[p_e1:\.Ltmp[0-9]+]]:
; ASM: .LBB0_3:                                # %if.else
; ASM:         callq   barrier
; ASM: .LBB0_4:                                # %if.end6
; ASM:         xorl    %eax, %eax
; ASM:         addq    $32, %rsp
; ASM:         popq    %rsi
; ASM:         retq
; ASM: .LBB0_5:                                # %if.then4
; ASM: [[p_b2:\.Ltmp[0-9]+]]:
; ASM:         #DEBUG_VALUE: p <- $esi
; ASM:         callq   call_noreturn
; ASM:         ud2
; ASM: .Lfunc_end0:

; ASM:         .short  {{.*}}         # Record length
; ASM:         .short  4414                    # Record kind: S_LOCAL
; ASM:         .long   116                     # TypeIndex
; ASM:         .short  0                       # Flags
; ASM:         .asciz  "p"
; ASM:         .cv_def_range    [[p_b1]] [[p_e1]] [[p_b2]] .Lfunc_end0, "A\021\027\000\000\000"
; ASM:         .short  2                       # Record length
; ASM:         .short  4431                    # Record kind: S_PROC_ID_END

; OBJ:         LocalSym {
; OBJ:           Type: int (0x74)
; OBJ:           VarName: p
; OBJ:         }
; OBJ-NOT:     LocalSym {
; OBJ:         DefRangeRegisterSym {
; OBJ-NEXT:      Kind:
; OBJ-NEXT:      Register: ESI (0x17)
; OBJ-NEXT:      MayHaveNoName: 0
; OBJ-NEXT:      LocalVariableAddrRange {
; OBJ-NEXT:        OffsetStart: .text+0x{{.*}}
; OBJ-NEXT:        ISectStart: 0x0
; OBJ-NEXT:        Range: 0x{{.*}}
; OBJ-NEXT:      }
; OBJ-NEXT:      LocalVariableAddrGap [
; OBJ-NEXT:        GapStartOffset: 0x{{.*}}
; OBJ-NEXT:        Range: 0x{{.*}}
; OBJ-NEXT:      ]
; OBJ-NEXT:    }

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24210"

; Function Attrs: nounwind uwtable
define i32 @f() local_unnamed_addr #0 !dbg !7 {
entry:
  %call = tail call i32 bitcast (i32 (...)* @barrier to i32 ()*)() #4, !dbg !15
  %tobool = icmp eq i32 %call, 0, !dbg !15
  br i1 %tobool, label %if.else, label %if.then, !dbg !16

if.then:                                          ; preds = %entry
  %call1 = tail call i32 bitcast (i32 (...)* @vardef to i32 ()*)() #4, !dbg !17
  tail call void @llvm.dbg.value(metadata i32 %call1, metadata !12, metadata !18), !dbg !19
  %call2 = tail call i32 bitcast (i32 (...)* @barrier to i32 ()*)() #4, !dbg !20
  %tobool3 = icmp eq i32 %call2, 0, !dbg !20
  br i1 %tobool3, label %if.end, label %if.then4, !dbg !22

if.then4:                                         ; preds = %if.then
  tail call void @call_noreturn(i32 %call1) #5, !dbg !23
  unreachable, !dbg !23

if.end:                                           ; preds = %if.then
  tail call void @use(i32 %call1) #4, !dbg !24
  br label %if.end6, !dbg !25

if.else:                                          ; preds = %entry
  %call5 = tail call i32 bitcast (i32 (...)* @barrier to i32 ()*)() #4, !dbg !26
  br label %if.end6

if.end6:                                          ; preds = %if.else, %if.end
  ret i32 0, !dbg !28
}

declare i32 @barrier(...) local_unnamed_addr #1

declare i32 @vardef(...) local_unnamed_addr #1

; Function Attrs: noreturn
declare void @call_noreturn(i32) local_unnamed_addr #2

declare void @use(i32) local_unnamed_addr #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { noreturn "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readnone }
attributes #4 = { nounwind }
attributes #5 = { noreturn nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.cpp", directory: "C:\5Csrc\5Cllvm\5Cbuild")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 4.0.0 "}
!7 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 19, type: !8, isLocal: false, isDefinition: true, scopeLine: 19, isOptimized: true, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "p", scope: !13, file: !1, line: 21, type: !10)
!13 = distinct !DILexicalBlock(scope: !14, file: !1, line: 20, column: 18)
!14 = distinct !DILexicalBlock(scope: !7, file: !1, line: 20, column: 7)
!15 = !DILocation(line: 20, column: 7, scope: !14)
!16 = !DILocation(line: 20, column: 7, scope: !7)
!17 = !DILocation(line: 21, column: 13, scope: !13)
!18 = !DIExpression()
!19 = !DILocation(line: 21, column: 9, scope: !13)
!20 = !DILocation(line: 22, column: 9, scope: !21)
!21 = distinct !DILexicalBlock(scope: !13, file: !1, line: 22, column: 9)
!22 = !DILocation(line: 22, column: 9, scope: !13)
!23 = !DILocation(line: 23, column: 7, scope: !21)
!24 = !DILocation(line: 24, column: 5, scope: !13)
!25 = !DILocation(line: 25, column: 3, scope: !13)
!26 = !DILocation(line: 26, column: 5, scope: !27)
!27 = distinct !DILexicalBlock(scope: !14, file: !1, line: 25, column: 10)
!28 = !DILocation(line: 28, column: 3, scope: !7)
