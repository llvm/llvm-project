; RUN: llc -mtriple=x86_64-linux-gnu -filetype=asm %s -o - | FileCheck %s

; The prologue-end line record must be emitted after the last instruction that
; is part of the function frame setup code and before the instruction that marks
; the beginning of the function body.
;
; For the given test, generated from:
;
; 1 extern int get_arg();
; 2 extern void func(int x);
; 3
; 4 int main()
; 5 {
; 6   int a;
; 7   func(get_arg());
; 8 }
; 9

; The prologue-end line record is emitted with an incorrect associated address,
; which causes a debugger to show the beginning of function body to be inside
; the prologue.

; This can be seen in the following trimmed assembler output:
;
; main:
;   ...
; # %bb.0:
;   .loc	1 7 0 prologue_end
;   pushq	%rax
;   .cfi_def_cfa_offset 16
;   callq	_Z7get_argv
;   ...
;   retq

; The instruction 'pushq %rax' is part of the frame setup code.

; The correct location for the prologue-end line information is just before
; the call to '_Z7get_argv', as illustrated in the following trimmed
; assembler output:
;
; main:
;   ...
; # %bb.0:
;   pushq  %rax
;   .cfi_def_cfa_offset 16
;   .loc  1 7 0 prologue_end
;   callq  _Z7get_argv
;   ...
;   retq

; Check that the generated assembler matches the following sequence:

; CHECK:      # %bb.0:
; CHECK-NEXT:  	pushq	%rax
; CHECK-NEXT:  	.cfi_def_cfa_offset 16
; CHECK-NEXT: .Ltmp0:
; CHECK-NEXT:  	.loc	1 7 8 prologue_end {{.*}}# invalid-prologue-end.cpp:7:8
; CHECK-NEXT:  	callq	_Z7get_argv

define i32 @main() #0 !dbg !7 {
entry:
  %a = alloca i32, align 4
  call void @llvm.dbg.declare(metadata ptr %a, metadata !11, metadata !DIExpression()), !dbg !12
  %call = call i32 @_Z7get_argv(), !dbg !13
  call void @_Z4funci(i32 %call), !dbg !14
  ret i32 0, !dbg !15
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1
declare void @_Z4funci(i32) #2
declare i32 @_Z7get_argv() #2

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 7.0.0 (trunk 322269)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "invalid-prologue-end.cpp", directory: "/home/carlos/llvm-root/work")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 7.0.0 (trunk 322269)"}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 4, type: !8, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocalVariable(name: "a", scope: !7, file: !1, line: 6, type: !10)
!12 = !DILocation(line: 6, column: 7, scope: !7)
!13 = !DILocation(line: 7, column: 8, scope: !7)
!14 = !DILocation(line: 7, column: 3, scope: !7)
!15 = !DILocation(line: 8, column: 1, scope: !7)

