; RUN: llc  -march=mipsel -mcpu=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=C1
; RUN: llc  -march=mipsel -mcpu=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=C2
; RUN: llc  -march=mipsel -mcpu=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=PE
;
; re-enable this when mips16's jalr is fixed.
; DISABLED: llc  -march=mipsel -mcpu=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=SR


@.str = private unnamed_addr constant [13 x i8] c"hello world\0A\00", align 1

define i32 @main() nounwind {
entry:
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([13 x i8]* @.str, i32 0, i32 0))
  ret i32 0

; SR: 	.set	mips16                  # @main

; SR:	save 	$ra, [[FS:[0-9]+]]
; PE:	li	$[[T1:[0-9]+]], %hi(_gp_disp)
; PE: 	addiu	$[[T2:[0-9]+]], $pc, %lo(_gp_disp)
; PE:	sll	$[[T3:[0-9]+]], $[[T1]], 16
; C1:	lw	${{[0-9]+}}, %got($.str)(${{[0-9]+}})
; C2:	lw	${{[0-9]+}}, %call16(printf)(${{[0-9]+}})
; C1:	addiu	${{[0-9]+}}, %lo($.str)
; C2:	move	$25, ${{[0-9]+}}
; C1:	move 	$gp, ${{[0-9]+}}
; C1:	jalrc 	${{[0-9]+}}
; SR:	restore 	$ra, [[FS]]
; PE:	li	$2, 0
; PE:	jrc 	$ra

}

declare i32 @printf(i8*, ...)
