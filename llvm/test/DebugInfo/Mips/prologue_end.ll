; RUN: llc -O0 -mtriple mips-unknown-linux-gnu -relocation-model=static < %s | FileCheck %s -check-prefix=STATIC
; RUN: llc -O0 -mtriple mips-unknown-linux-gnu -relocation-model=static -frame-pointer=all < %s | FileCheck %s -check-prefix=STATIC-FP
; RUN: llc -O0 -mtriple mips-unknown-linux-gnu -relocation-model=pic < %s | FileCheck %s -check-prefix=PIC
; RUN: llc -O0 -mtriple mips-unknown-linux-gnu -relocation-model=pic -frame-pointer=all < %s | FileCheck %s -check-prefix=PIC-FP

; Generated using clang -O0 -emit-llvm -S -target mipsel-unknown-linux -g test.c -o test.ll
; test.c:
;
; void hello_world(void) {
;   printf("Hello, World!\n");
; }

@.str = private unnamed_addr constant [15 x i8] c"Hello, World!\0A\00", align 1

define void @hello_world() #0 !dbg !4 {
entry:
; STATIC:	addiu	$sp, $sp, -{{[0-9]+}}
; STATIC:	sw	$ra, {{[0-9]+}}($sp)
; STATIC:	.loc	1 2 3 prologue_end
; STATIC:	lui	$[[R0:[0-9]+]], %hi($.str)

; STATIC-FP:	addiu	$sp, $sp, -{{[0-9]+}}
; STATIC-FP:	sw	$ra, {{[0-9]+}}($sp)
; STATIC-FP:	sw	$fp, {{[0-9]+}}($sp)
; STATIC-FP:	move	$fp, $sp
; STATIC-FP:	.loc	1 2 3 prologue_end
; STATIC-FP:	lui	$[[R0:[0-9]+]], %hi($.str)

; PIC:     	lui	$[[R0:[0-9]+]], %hi(_gp_disp)
; PIC:     	addiu	$[[R0]], $[[R0]], %lo(_gp_disp)
; PIC:     	addiu	$sp, $sp, -{{[0-9]+}}
; PIC:     	sw	$ra, {{[0-9]+}}($sp)
; PIC:     	addu	$[[R1:[0-9]+|gp]], $[[R0]], $25
; PIC:     	.loc	1 2 3 prologue_end
; PIC:     	lw	$[[R2:[0-9]+]], %got($.str)($[[R1]])

; PIC-FP:	lui	$[[R0:[0-9]+]], %hi(_gp_disp)
; PIC-FP:	addiu	$[[R0]], $[[R0]], %lo(_gp_disp)
; PIC-FP:	addiu	$sp, $sp, -{{[0-9]+}}
; PIC-FP:	sw	$ra, {{[0-9]+}}($sp)
; PIC-FP:	sw	$fp, {{[0-9]+}}($sp)
; PIC-FP:	move	$fp, $sp
; PIC-FP:	addu	$[[R1:[0-9]+|gp]], $[[R0]], $25
; PIC-FP:	.loc	1 2 3 prologue_end
; PIC-FP:	lw	$[[R2:[0-9]+]], %got($.str)($[[R1]])

  %call = call i32 (ptr, ...) @printf(ptr @.str), !dbg !10
  ret void, !dbg !11
}

declare i32 @printf(ptr, ...)

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !{}
!4 = distinct !DISubprogram(name: "hello_world", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang version 3.8.0"}
!10 = !DILocation(line: 2, column: 3, scope: !4)
!11 = !DILocation(line: 3, column: 1, scope: !4)
