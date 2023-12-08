; RUN: llc  -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=C1
; RUN: llc  -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=C2
; RUN: llc  -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=PE
; RUN: llc  -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -relocation-model=static -O3 < %s | FileCheck %s -check-prefix=ST1
; RUN: llc  -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -relocation-model=static -O3 < %s | FileCheck %s -check-prefix=ST2
;
; RUN: llc  -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=SR
; RUN: llc  -mtriple=mipsel-linux-gnu -march=mipsel -mcpu=mips32  -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=SR32


@.str = private unnamed_addr constant [13 x i8] c"hello world\0A\00", align 1

define i32 @main() nounwind {
entry:
  %call = call i32 (ptr, ...) @printf(ptr @.str)
  ret i32 0

; SR: 	.set	mips16

; SR32: .set nomips16
; SR32: .ent main
; SR-NOT:  .set noreorder
; SR-NOT:  .set nomacro
; SR-NOT:  .set noat
; SR32:  .set noreorder
; SR32:  .set nomacro
; SR32:  .set noat
; SR:	save 	$ra, 24 # 16 bit inst
; PE:    .ent main
; PE:	li	$[[T1:[0-9]+]], %hi(_gp_disp)
; PE-NEXT: 	addiu	$[[T2:[0-9]+]], $pc, %lo(_gp_disp)
; PE:	        sll	$[[T3:[0-9]+]], $[[T1]], 16
; C1:	lw	${{[0-9]+}}, %got($.str)(${{[0-9]+}})
; C2:	lw	${{[0-9]+}}, %call16(printf)(${{[0-9]+}})
; C1:	addiu	${{[0-9]+}}, %lo($.str)
; C2:	move	$25, ${{[0-9]+}}
; C1:	move 	$gp, ${{[0-9]+}}
; C1:	jalrc 	${{[0-9]+}}
; SR:	restore $ra,	24 # 16 bit inst
; PE:	li	$2, 0
; PE:	jrc 	$ra

; ST1:  li	${{[0-9]+}}, %hi($.str)
; ST1:  sll     ${{[0-9]+}}, ${{[0-9]+}}, 16
; ST1:	addiu	${{[0-9]+}}, %lo($.str)
; ST2:  li	${{[0-9]+}}, %hi($.str)
; ST2:  jal     printf
}

;  SR-NOT:  .set at
;  SR-NOT:  .set macro
;  SR-NOT:  .set reorder
;  SR32:  .set at
;  SR32:  .set macro
;  SR32:  .set reorder
; SR:   .end main
; SR32:   .end main
declare i32 @printf(ptr, ...)
