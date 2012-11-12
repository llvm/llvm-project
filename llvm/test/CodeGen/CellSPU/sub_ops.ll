; RUN: llc < %s -march=cellspu | FileCheck %s

define i32 @subword( i32 %param1, i32 %param2) {
; Check ordering of registers ret=param1-param2 -> rt=rb-ra
; CHECK-NOT:	sf	$3, $3, $4
; CHECK:	sf	$3, $4, $3
	%1 = sub i32 %param1, %param2
	ret i32 %1
}

define i16 @subhword( i16 %param1, i16 %param2) {
; Check ordering of registers ret=param1-param2 -> rt=rb-ra
; CHECK-NOT:	sfh	$3, $3, $4
; CHECK:	sfh	$3, $4, $3
	%1 = sub i16 %param1, %param2
	ret i16 %1
}

define float @subfloat( float %param1, float %param2) {
; Check ordering of registers ret=param1-param2 -> rt=ra-rb 
; (yes this is reverse of i32 instruction)
; CHECK-NOT:	fs	$3, $4, $3 
; CHECK:	fs	$3, $3, $4
	%1 = fsub float %param1, %param2
	ret float %1
}
