; XFAIL: *
; ...should pass. See PR12324: misched bringup
; RUN: llc < %s -march=x86-64 | FileCheck %s
; <rdar://problem/8006248>

@llvm.used = appending global [1 x i8*] [i8* bitcast (void ([40 x i16]*, i32*, i16**, i64*)* @func to i8*)], section "llvm.metadata"

define void @func([40 x i16]* %a, i32* %b, i16** %c, i64* %d) nounwind {
entry:
  %tmp103 = getelementptr inbounds [40 x i16]* %a, i64 0, i64 4
  %tmp104 = load i16* %tmp103, align 2
  %tmp105 = sext i16 %tmp104 to i32
  %tmp106 = load i32* %b, align 4
  %tmp107 = sub nsw i32 4, %tmp106
  %tmp108 = load i16** %c, align 8
  %tmp109 = sext i32 %tmp107 to i64
  %tmp110 = getelementptr inbounds i16* %tmp108, i64 %tmp109
  %tmp111 = load i16* %tmp110, align 1
  %tmp112 = sext i16 %tmp111 to i32
  %tmp = mul i32 355244649, %tmp112
  %tmp1 = mul i32 %tmp, %tmp105
  %tmp2 = add i32 %tmp1, 2138875574
  %tmp3 = add i32 %tmp2, 1546991088
  %tmp4 = mul i32 %tmp3, 2122487257
  %tmp5 = icmp sge i32 %tmp4, 2138875574
  %tmp6 = icmp slt i32 %tmp4, -8608074
  %tmp7 = or i1 %tmp5, %tmp6
  %outSign = select i1 %tmp7, i32 1, i32 -1
  %tmp8 = icmp slt i32 %tmp4, 0
  %tmp9 = icmp eq i32 %outSign, 1
  %tmp10 = and i1 %tmp8, %tmp9
  %tmp11 = sext i32 %tmp4 to i64
  %tmp12 = add i64 %tmp11, 5089792279245435153

; CHECK:      addl	$2138875574, %e[[REGISTER_zext:[a-z]+]]
; CHECK-NEXT: movslq	%e[[REGISTER_zext]], [[REGISTER_tmp:%[a-z]+]]
; CHECK:      movq	[[REGISTER_tmp]], [[REGISTER_sext:%[a-z]+]]
; CHECK-NEXT: subq	%r[[REGISTER_zext]], [[REGISTER_sext]]

  %tmp13 = sub i64 %tmp12, 2138875574
  %tmp14 = zext i32 %tmp4 to i64
  %tmp15 = sub i64 %tmp11, %tmp14
  %tmp16 = select i1 %tmp10, i64 %tmp15, i64 0
  %tmp17 = sub i64 %tmp13, %tmp16
  %tmp18 = mul i64 %tmp17, 4540133155013554595
  %tmp19 = sub i64 %tmp18, 5386586244038704851
  %tmp20 = add i64 %tmp19, -1368057358110947217
  %tmp21 = mul i64 %tmp20, -422037402840850817
  %tmp115 = load i64* %d, align 8
  %alphaX = mul i64 468858157810230901, %tmp21
  %alphaXbetaY = add i64 %alphaX, %tmp115
  %transformed = add i64 %alphaXbetaY, 9040145182981852475
  store i64 %transformed, i64* %d, align 8
  ret void
}
