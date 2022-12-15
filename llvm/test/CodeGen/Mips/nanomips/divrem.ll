; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

; Make sure to generate __udivmoddi4 libcall when udiv and urem 
; instructions with the same operands are present 
; and the operands are of type int64
define void @test1(i64 %a, i64 %b, i64* %divmod) {    
  ; CHECK: save	16, $ra, $s0
  ; CHECK: move	$s0, $a4
  ; CHECK: move	$a4, $sp
  ; CHECK: balc	__udivmoddi4
  ; CHECK: swm	$a0, 0($s0), 2
  ; CHECK: lw	$a0, 4($sp)
  ; CHECK: sw	$a0, 12($s0)
  ; CHECK: lw	$a0, 0($sp)
  ; CHECK: sw	$a0, 8($s0)
  ; CHECK: restore.jrc	16, $ra, $s0
  %div = udiv i64 %a, %b
  store i64 %div, i64* %divmod, align 8
  %rem = urem i64 %a, %b
  %arrayidx1 = getelementptr inbounds i64, i64* %divmod, i32 1
  store i64 %rem, i64* %arrayidx1, align 8
  ret void
}

; Make sure to generate __umoddi3 libcall when only urem is present
; and the operands are of type int64
define void @test2(i64 %a, i64 %b, i64* %divmod) {
	; CHECK: save	16, $ra, $s0
	; CHECK: move	$s0, $a4
	; CHECK: balc	__umoddi3
	; CHECK: swm	$a0, 8($s0), 2
	; CHECK: restore.jrc	16, $ra, $s0
  %rem = urem i64 %a, %b
  %arrayidx = getelementptr inbounds i64, i64* %divmod, i32 1
  store i64 %rem, i64* %arrayidx, align 8
  ret void
}

; Make sure to generate __udivdi3 libcall when only udiv is present
; and the operands are of type int64
define void @test3(i64 %a, i64 %b, i64* %divmod) {
	; CHECK: save	16, $ra, $s0
	; CHECK: move	$s0, $a4
	; CHECK: balc	__udivdi3
	; CHECK: swm	$a0, 0($s0), 2
	; CHECK: restore.jrc	16, $ra, $s0
  %div = udiv i64 %a, %b
  store i64 %div, i64* %divmod, align 8
  ret void
}

; If urem is expanded into mul+sub and the operands 
; are of type int64, make sure to stay that way
define void @test4(i64 %a, i64 %b, i64* %divmod) {
  ; CHECK: save	32, $ra, $s0, $s1, $s2, $s3, $s4
	; CHECK: movep	$s1, $s0, $a3, $a4
	; CHECK: movep	$s4, $s2, $a1, $a2
	; CHECK: move	$s3, $a0
	; CHECK: balc	__udivdi3
	; CHECK: mul	$a2, $a0, $s2
	; CHECK: subu	$a3, $s3, $a2
	; CHECK: sw	$a3, 8($s0)
	; CHECK: mul	$a3, $a0, $s1
	; CHECK: muhu	$s1, $a0, $s2
	; CHECK: addu	$a3, $s1, $a3
	; CHECK: swm	$a0, 0($s0), 2
	; CHECK: mul	$a0, $a1, $s2
	; CHECK: addu	$a0, $a3, $a0
	; CHECK: subu	$a0, $s4, $a0
	; CHECK: sltu	$a1, $s3, $a2
	; CHECK: subu	$a0, $a0, $a1
	; CHECK: sw	$a0, 12($s0)
	; CHECK: restore.jrc	32, $ra, $s0, $s1, $s2, $s3, $s4
  %a.frozen = freeze i64 %a
  %b.frozen = freeze i64 %b
  %div = udiv i64 %a.frozen, %b.frozen
  store i64 %div, i64* %divmod, align 8
  %1 = mul i64 %div, %b.frozen
  %rem.decomposed = sub i64 %a.frozen, %1
  %arrayidx1 = getelementptr inbounds i64, i64* %divmod, i32 1
  store i64 %rem.decomposed, i64* %arrayidx1, align 8
  ret void
}

; Make sure to generate divu and modu when udiv and urem 
; instructions with the same operands are present 
; and the operands are of type int32
define void @test5(i32 %a, i32 %b, i32* %divmod) {
  ; CHECK: modu	$a3, $a0, $a1
	; CHECK: teq	$zero, $a1, 7
	; CHECK: sw	$a3, 4($a2)
	; CHECK: divu	$a0, $a0, $a1
	; CHECK: teq	$zero, $a1, 7
	; CHECK: sw	$a0, 0($a2)
	; CHECK: jrc	$ra
  %div = udiv i32 %a, %b
  store i32 %div, i32* %divmod, align 4
  %rem = urem i32 %a, %b
  %arrayidx1 = getelementptr inbounds i32, i32* %divmod, i32 1
  store i32 %rem, i32* %arrayidx1, align 4
  ret void
}

; Make sure to generate modu when only urem is present
; and the operands are of type int32
define  void @test6(i32 %a, i32 %b, i32* %divmod) {
  ; CHECK: modu	$a0, $a0, $a1
	; CHECK: teq	$zero, $a1, 7
	; CHECK: sw	$a0, 4($a2)
	; CHECK: jrc	$ra
  %rem = urem i32 %a, %b
  %arrayidx = getelementptr inbounds i32, i32* %divmod, i32 1
  store i32 %rem, i32* %arrayidx, align 4
  ret void
}

; Make sure to generate divu when only udiv is present
; and the operands are of type int32
define void @test7(i32 %a, i32 %b, i32* %divmod) {
  ; CHECK: divu	$a0, $a0, $a1
	; CHECK: teq	$zero, $a1, 7
	; CHECK: sw	$a0, 0($a2)
	; CHECK: jrc	$ra
  %div = udiv i32 %a, %b
  store i32 %div, i32* %divmod, align 4
  ret void
}

; If urem is expanded into mul+sub and the operands 
; are of type int32, make sure to stay that way.
define void @test8(i32 %a, i32 %b, i32* %divmod) {
  ; CHECK: divu	$a3, $a0, $a1
	; CHECK: teq	$zero, $a1, 7
	; CHECK: sw	$a3, 0($a2)
	; CHECK: mul	$a1, $a3, $a1
	; CHECK: subu	$a0, $a0, $a1
	; CHECK: sw	$a0, 4($a2)
	; CHECK: jrc	$ra
  %a.frozen = freeze i32 %a
  %b.frozen = freeze i32 %b
  %div = udiv i32 %a.frozen, %b.frozen
  store i32 %div, i32* %divmod, align 4
  %1 = mul i32 %div, %b.frozen
  %rem.decomposed = sub i32 %a.frozen, %1
  %arrayidx1 = getelementptr inbounds i32, i32* %divmod, i32 1
  store i32 %rem.decomposed, i32* %arrayidx1, align 4
  ret void
}

