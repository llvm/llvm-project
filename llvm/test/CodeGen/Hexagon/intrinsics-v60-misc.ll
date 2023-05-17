; RUN: llc -march=hexagon < %s | FileCheck %s

@l = external global <32 x i32>
@k = external global <16 x i32>
@h = external global <16 x i32>
@n = external global i64
@m = external global i32

; CHECK-LABEL: test1:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w = vrmpy(v{{[0-9]+}}:{{[0-9]+}}.ub,r{{[0-9]+}}.b,#1)
define void @test1(<32 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi(<32 x i32> %a, i32 %b, i32 1)
  store <32 x i32> %0, ptr @l, align 128
  ret void
}

; CHECK-LABEL: test2:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.uw = vrsad(v{{[0-9]+}}:{{[0-9]+}}.ub,r{{[0-9]+}}.ub,#1)
define void @test2(<32 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vrsadubi(<32 x i32> %a, i32 %b, i32 1)
  store <32 x i32> %0, ptr @l, align 128
  ret void
}

; CHECK-LABEL: test3:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.uw = vrmpy(v{{[0-9]+}}:{{[0-9]+}}.ub,r{{[0-9]+}}.ub,#1)
define void @test3(<32 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vrmpyubi(<32 x i32> %a, i32 %b, i32 1)
  store <32 x i32> %0, ptr @l, align 128
  ret void
}

; CHECK-LABEL: test4:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w += vrmpy(v{{[0-9]+}}:{{[0-9]+}}.ub,r{{[0-9]+}}.b,#1)
define void @test4(<32 x i32> %a, <32 x i32> %b, i32 %c) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32> %a, <32 x i32> %b, i32 %c, i32 1)
  store <32 x i32> %0, ptr @l, align 128
  ret void
}

; CHECK-LABEL: test5:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.uw += vrsad(v{{[0-9]+}}:{{[0-9]+}}.ub,r{{[0-9]+}}.ub,#1)
define void @test5(<32 x i32> %a, <32 x i32> %b, i32 %c) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vrsadubi.acc(<32 x i32> %a, <32 x i32> %b, i32 %c, i32 1)
  store <32 x i32> %0, ptr @l, align 128
  ret void
}

; CHECK-LABEL: test6:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.uw += vrmpy(v{{[0-9]+}}:{{[0-9]+}}.ub,r{{[0-9]+}}.ub,#0)
define void @test6(<32 x i32> %a, <32 x i32> %b, i32 %c) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vrmpyubi.acc(<32 x i32> %a, <32 x i32> %b, i32 %c, i32 0)
  store <32 x i32> %0, ptr @l, align 128
  ret void
}

; CHECK-LABEL: test7:
; CHECK: v{{[0-9]+}} = valign(v{{[0-9]+}},v{{[0-9]+}},r{{[0-9]+}})
define void @test7(<16 x i32> %a, <16 x i32> %b, i32 %c) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %a, <16 x i32> %b, i32 %c)
  store <16 x i32> %0, ptr @k, align 64
  ret void
}

; CHECK-LABEL: test8:
; CHECK: v{{[0-9]+}} = vlalign(v{{[0-9]+}},v{{[0-9]+}},r{{[0-9]+}})
define void @test8(<16 x i32> %a, <16 x i32> %b, i32 %c) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vlalignb(<16 x i32> %a, <16 x i32> %b, i32 %c)
  store <16 x i32> %0, ptr @k, align 64
  ret void
}

; CHECK-LABEL: test9:
; CHECK: v{{[0-9]+}}.h = vasr(v{{[0-9]+}}.w,v{{[0-9]+}}.w,r{{[0-9]+}})
define void @test9(<16 x i32> %a, <16 x i32> %b, i32 %c) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vasrwh(<16 x i32> %a, <16 x i32> %b, i32 %c)
  store <16 x i32> %0, ptr @k, align 64
  ret void
}

; CHECK-LABEL: test10:
; CHECK: v{{[0-9]+}}.h = vasr(v{{[0-9]+}}.w,v{{[0-9]+}}.w,r{{[0-9]+}}):sat
define void @test10(<16 x i32> %a, <16 x i32> %b, i32 %c) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vasrwhsat(<16 x i32> %a, <16 x i32> %b, i32 %c)
  store <16 x i32> %0, ptr @k, align 64
  ret void
}

; CHECK-LABEL: test11:
; CHECK: v{{[0-9]+}}.h = vasr(v{{[0-9]+}}.w,v{{[0-9]+}}.w,r{{[0-9]+}}):rnd:sat
define void @test11(<16 x i32> %a, <16 x i32> %b, i32 %c) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vasrwhrndsat(<16 x i32> %a, <16 x i32> %b, i32 %c)
  store <16 x i32> %0, ptr @k, align 64
  ret void
}

; CHECK-LABEL: test12:
; CHECK: v{{[0-9]+}}.uh = vasr(v{{[0-9]+}}.w,v{{[0-9]+}}.w,r{{[0-9]+}}):sat
define void @test12(<16 x i32> %a, <16 x i32> %b, i32 %c) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vasrwuhsat(<16 x i32> %a, <16 x i32> %b, i32 %c)
  store <16 x i32> %0, ptr @k, align 64
  ret void
}

; CHECK-LABEL: test13:
; CHECK: v{{[0-9]+}}.ub = vasr(v{{[0-9]+}}.h,v{{[0-9]+}}.h,r{{[0-9]+}}):sat
define void @test13(<16 x i32> %a, <16 x i32> %b, i32 %c) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vasrhubsat(<16 x i32> %a, <16 x i32> %b, i32 %c)
  store <16 x i32> %0, ptr @k, align 64
  ret void
}

; CHECK-LABEL: test14:
; CHECK: v{{[0-9]+}}.ub = vasr(v{{[0-9]+}}.h,v{{[0-9]+}}.h,r{{[0-9]+}}):rnd:sat
define void @test14(<16 x i32> %a, <16 x i32> %b, i32 %c) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vasrhubrndsat(<16 x i32> %a, <16 x i32> %b, i32 %c)
  store <16 x i32> %0, ptr @k, align 64
  ret void
}

; CHECK-LABEL: test15:
; CHECK: v{{[0-9]+}}.b = vasr(v{{[0-9]+}}.h,v{{[0-9]+}}.h,r{{[0-9]+}}):rnd:sat
define void @test15(<16 x i32> %a, <16 x i32> %b, i32 %c) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vasrhbrndsat(<16 x i32> %a, <16 x i32> %b, i32 %c)
  store <16 x i32> %0, ptr @k, align 64
  ret void
}

; CHECK-LABEL: test16:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h |= vunpacko(v{{[0-9]+}}.b)
define void @test16(<32 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vunpackob(<32 x i32> %a, <16 x i32> %b)
  store <32 x i32> %0, ptr @l, align 128
  ret void
}

; CHECK-LABEL: test17:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w |= vunpacko(v{{[0-9]+}}.h)
define void @test17(<32 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vunpackoh(<32 x i32> %a, <16 x i32> %b)
  store <32 x i32> %0, ptr @l, align 128
  ret void
}

; CHECK-LABEL: test18:
; CHECK: v{{[0-9]+}} = valign(v{{[0-9]+}},v{{[0-9]+}},#3)
define void @test18(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %a, <16 x i32> %b, i32 3)
  store <16 x i32> %0, ptr @k, align 64
  ret void
}

; CHECK-LABEL: test19:
; CHECK: v{{[0-9]+}} = vlalign(v{{[0-9]+}},v{{[0-9]+}},#3)
define void @test19(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %a, <16 x i32> %b, i32 3)
  store <16 x i32> %0, ptr @k, align 64
  ret void
}

; CHECK-LABEL: test20:
; CHECK: v{{[0-9]+}} = vmux(q{{[0-3]+}},v{{[0-9]+}},v{{[0-9]+}})
define void @test20(<16 x i32> %a, <16 x i32> %b, <16 x i32> %c) #0 {
entry:
  %0 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %a, i32 -1)
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vmux(<64 x i1> %0, <16 x i32> %b, <16 x i32> %c)
  store <16 x i32> %1, ptr @k, align 64
  ret void
}

; CHECK-LABEL: test21:
; CHECK: q{{[0-3]+}} = and(q{{[0-3]+}},q{{[0-3]+}})
define void @test21(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %a, i32 -1)
  %1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %b, i32 -1)
  %2 = tail call <64 x i1> @llvm.hexagon.V6.pred.and(<64 x i1> %0, <64 x i1> %1)
  %3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %2, i32 -1)
  store <16 x i32> %3, ptr @h, align 64
  ret void
}

; CHECK-LABEL: test22:
; CHECK: q{{[0-3]+}} = or(q{{[0-3]+}},q{{[0-3]+}})
define void @test22(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %a, i32 -1)
  %1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %b, i32 -1)
  %2 = tail call <64 x i1> @llvm.hexagon.V6.pred.or(<64 x i1> %0, <64 x i1> %1)
  %3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %2, i32 -1)
  store <16 x i32> %3, ptr @h, align 64
  ret void
}

; CHECK-LABEL: test23:
; CHECK: q{{[0-3]+}} = not(q{{[0-3]+}})
define void @test23(<16 x i32> %a) #0 {
entry:
  %0 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %a, i32 -1)
  %1 = tail call <64 x i1> @llvm.hexagon.V6.pred.not(<64 x i1> %0)
  %2 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %1, i32 -1)
  store <16 x i32> %2, ptr @h, align 64
  ret void
}

; CHECK-LABEL: test24:
; CHECK: q{{[0-3]+}} = xor(q{{[0-3]+}},q{{[0-3]+}})
define void @test24(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %a, i32 -1)
  %1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %b, i32 -1)
  %2 = tail call <64 x i1> @llvm.hexagon.V6.pred.xor(<64 x i1> %0, <64 x i1> %1)
  %3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %2, i32 -1)
  store <16 x i32> %3, ptr @h, align 64
  ret void
}

; CHECK-LABEL: test25:
; CHECK: q{{[0-3]+}} = or(q{{[0-3]+}},!q{{[0-3]+}})
define void @test25(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %a, i32 -1)
  %1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %b, i32 -1)
  %2 = tail call <64 x i1> @llvm.hexagon.V6.pred.or.n(<64 x i1> %0, <64 x i1> %1)
  %3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %2, i32 -1)
  store <16 x i32> %3, ptr @h, align 64
  ret void
}

; CHECK-LABEL: test26:
; CHECK: q{{[0-3]+}} = and(q{{[0-3]+}},!q{{[0-3]+}})
define void @test26(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %a, i32 -1)
  %1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %b, i32 -1)
  %2 = tail call <64 x i1> @llvm.hexagon.V6.pred.and.n(<64 x i1> %0, <64 x i1> %1)
  %3 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %2, i32 -1)
  store <16 x i32> %3, ptr @h, align 64
  ret void
}

; CHECK-LABEL: test27:
; CHECK: q{{[0-3]+}} = vcmp.gt(v{{[0-9]+}}.ub,v{{[0-9]+}}.ub)
define void @test27(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <64 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> %a, <16 x i32> %b)
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %0, i32 -1)
  store <16 x i32> %1, ptr @k, align 64
  ret void
}

; CHECK-LABEL: test28:
; CHECK: q{{[0-3]+}} = vcmp.gt(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define void @test28(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <64 x i1> @llvm.hexagon.V6.vgth(<16 x i32> %a, <16 x i32> %b)
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %0, i32 -1)
  store <16 x i32> %1, ptr @k, align 64
  ret void
}

; CHECK-LABEL: test29:
; CHECK: q{{[0-3]+}} = vcmp.eq(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define void @test29(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <64 x i1> @llvm.hexagon.V6.veqh(<16 x i32> %a, <16 x i32> %b)
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %0, i32 -1)
  store <16 x i32> %1, ptr @k, align 64
  ret void
}

; CHECK-LABEL: test30:
; CHECK: q{{[0-3]+}} = vcmp.gt(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
define void @test30(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <64 x i1> @llvm.hexagon.V6.vgtw(<16 x i32> %a, <16 x i32> %b)
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %0, i32 -1)
  store <16 x i32> %1, ptr @k, align 64
  ret void
}

; CHECK-LABEL: test31:
; CHECK: q{{[0-3]+}} = vcmp.eq(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
define void @test31(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <64 x i1> @llvm.hexagon.V6.veqw(<16 x i32> %a, <16 x i32> %b)
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %0, i32 -1)
  store <16 x i32> %1, ptr @k, align 64
  ret void
}

; CHECK-LABEL: test32:
; CHECK: q{{[0-3]+}} = vcmp.gt(v{{[0-9]+}}.uh,v{{[0-9]+}}.uh)
define void @test32(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <64 x i1> @llvm.hexagon.V6.vgtuh(<16 x i32> %a, <16 x i32> %b)
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %0, i32 -1)
  store <16 x i32> %1, ptr @k, align 64
  ret void
}

; CHECK-LABEL: test33:
; CHECK: v{{[0-9]+}} |= vand(q{{[0-3]+}},r{{[0-9]+}})
define void @test33(<16 x i32> %a, <16 x i32> %b, i32 %c) #0 {
entry:
  %0 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %b, i32 -1)
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt.acc(<16 x i32> %a, <64 x i1> %0, i32 %c)
  store <16 x i32> %1, ptr @h, align 64
  ret void
}

; CHECK-LABEL: test34:
; CHECK: q{{[0-3]+}} |= vand(v{{[0-9]+}},r{{[0-9]+}})
define void @test34(<16 x i32> %a, <16 x i32> %b, i32 %c) #0 {
entry:
  %0 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %a, i32 -1)
  %1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt.acc(<64 x i1> %0, <16 x i32> %b, i32 %c)
  %2 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %1, i32 -1)
  store <16 x i32> %2, ptr @k, align 64
  ret void
}

; CHECK-LABEL: test35:
; CHECK: v{{[0-9]+}} = vand(q{{[0-3]+}},r{{[0-9]+}})
define void @test35(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %a, i32 -1)
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %0, i32 %b)
  store <16 x i32> %1, ptr @h, align 64
  ret void
}

; CHECK-LABEL: test36:
; CHECK: q{{[0-3]+}} = vand(v{{[0-9]+}},r{{[0-9]+}})
define void @test36(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %a, i32 %b)
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %0, i32 -1)
  store <16 x i32> %1, ptr @k, align 64
  ret void
}

; CHECK-LABEL: test37:
; CHECK: r{{[0-9]+}}:{{[0-9]+}} = rol(r{{[0-9]+}}:{{[0-9]+}},#38)
define void @test37(i64 %a) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.S6.rol.i.p(i64 %a, i32 38)
  store i64 %0, ptr @n, align 8
  ret void
}

; CHECK-LABEL: test38:
; CHECK: r{{[0-9]+}}:{{[0-9]+}} += rol(r{{[0-9]+}}:{{[0-9]+}},#36)
define void @test38(i64 %a, i64 %b) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.S6.rol.i.p.acc(i64 %a, i64 %b, i32 36)
  store i64 %0, ptr @n, align 8
  ret void
}

; CHECK-LABEL: test39:
; CHECK: r{{[0-9]+}}:{{[0-9]+}} &= rol(r{{[0-9]+}}:{{[0-9]+}},#25)
define void @test39(i64 %a, i64 %b) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.S6.rol.i.p.and(i64 %a, i64 %b, i32 25)
  store i64 %0, ptr @n, align 8
  ret void
}

; CHECK-LABEL: test40:
; CHECK: r{{[0-9]+}}:{{[0-9]+}} -= rol(r{{[0-9]+}}:{{[0-9]+}},#20)
define void @test40(i64 %a, i64 %b) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.S6.rol.i.p.nac(i64 %a, i64 %b, i32 20)
  store i64 %0, ptr @n, align 8
  ret void
}

; CHECK-LABEL: test41:
; CHECK: r{{[0-9]+}}:{{[0-9]+}} |= rol(r{{[0-9]+}}:{{[0-9]+}},#22)
define void @test41(i64 %a, i64 %b) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.S6.rol.i.p.or(i64 %a, i64 %b, i32 22)
  store i64 %0, ptr @n, align 8
  ret void
}

; CHECK-LABEL: test42:
; CHECK: r{{[0-9]+}}:{{[0-9]+}} ^= rol(r{{[0-9]+}}:{{[0-9]+}},#25)
define void @test42(i64 %a, i64 %b) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.S6.rol.i.p.xacc(i64 %a, i64 %b, i32 25)
  store i64 %0, ptr @n, align 8
  ret void
}

; CHECK-LABEL: test43:
; CHECK: r{{[0-9]+}} = rol(r{{[0-9]+}},#14)
define void @test43(i32 %a) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.S6.rol.i.r(i32 %a, i32 14)
  %conv = sext i32 %0 to i64
  store i64 %conv, ptr @n, align 8
  ret void
}

; CHECK-LABEL: test44:
; CHECK: r{{[0-9]+}} += rol(r{{[0-9]+}},#12)
define void @test44(i32 %a, i32 %b) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.S6.rol.i.r.acc(i32 %a, i32 %b, i32 12)
  store i32 %0, ptr @m, align 4
  ret void
}

; CHECK-LABEL: test45:
; CHECK: r{{[0-9]+}} &= rol(r{{[0-9]+}},#18)
define void @test45(i32 %a, i32 %b) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.S6.rol.i.r.and(i32 %a, i32 %b, i32 18)
  store i32 %0, ptr @m, align 4
  ret void
}

; CHECK-LABEL: test46:
; CHECK: r{{[0-9]+}} -= rol(r{{[0-9]+}},#31)
define void @test46(i32 %a, i32 %b) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.S6.rol.i.r.nac(i32 %a, i32 %b, i32 31)
  store i32 %0, ptr @m, align 4
  ret void
}

; CHECK-LABEL: test47:
; CHECK: r{{[0-9]+}} |= rol(r{{[0-9]+}},#30)
define void @test47(i32 %a, i32 %b) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.S6.rol.i.r.or(i32 %a, i32 %b, i32 30)
  store i32 %0, ptr @m, align 4
  ret void
}

; CHECK-LABEL: test48:
; CHECK: r{{[0-9]+}} ^= rol(r{{[0-9]+}},#31)
define void @test48(i32 %a, i32 %b) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.S6.rol.i.r.xacc(i32 %a, i32 %b, i32 31)
  store i32 %0, ptr @m, align 4
  ret void
}

; CHECK-LABEL: test49:
; CHECK: r{{[0-9]+}} = vextract(v{{[0-9]+}},r{{[0-9]+}})
define void @test49(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.V6.extractw(<16 x i32> %a, i32 %b)
  store i32 %0, ptr @m, align 4
  ret void
}

; CHECK-LABEL: test50:
; CHECK: v{{[0-9]+}} = vsplat(r{{[0-9]+}})
define void @test50(i32 %a) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 %a)
  store <16 x i32> %0, ptr @k, align 64
  ret void
}

; CHECK-LABEL: test51:
; CHECK: q{{[0-3]}} = vsetq(r{{[0-9]+}})
define void @test51(i32 %a) #0 {
entry:
  %0 = tail call <64 x i1> @llvm.hexagon.V6.pred.scalar2(i32 %a)
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %0, i32 -1)
  store <16 x i32> %1, ptr @k, align 64
  ret void
}

; CHECK-LABEL: test52:
; CHECK: v{{[0-9]+}}.b = vlut32(v{{[0-9]+}}.b,v{{[0-9]+}}.b,r{{[0-9]+}})
define void @test52(<16 x i32> %a, <16 x i32> %b, i32 %c) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vlutvvb(<16 x i32> %a, <16 x i32> %b, i32 %c)
  store <16 x i32> %0, ptr @k, align 64
  ret void
}

; CHECK-LABEL: test53:
; CHECK: v{{[0-9]+}}.b |= vlut32(v{{[0-9]+}}.b,v{{[0-9]+}}.b,r{{[0-9]+}})
define void @test53(<16 x i32> %a, <16 x i32> %b, <16 x i32> %c, i32 %d) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vlutvvb.oracc(<16 x i32> %a, <16 x i32> %b, <16 x i32> %c, i32 %d)
  store <16 x i32> %0, ptr @k, align 64
  ret void
}

; CHECK-LABEL: test54:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h |= vlut16(v{{[0-9]+}}.b,v{{[0-9]+}}.h,r{{[0-9]+}})
define void @test54(<32 x i32> %a, <16 x i32> %b, <16 x i32> %c, i32 %d) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vlutvwh.oracc(<32 x i32> %a, <16 x i32> %b, <16 x i32> %c, i32 %d)
  store <32 x i32> %0, ptr @l, align 128
  ret void
}

; CHECK-LABEL: test55:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h = vlut16(v{{[0-9]+}}.b,v{{[0-9]+}}.h,r{{[0-9]+}})
define void @test55(<16 x i32> %a, <16 x i32> %b, i32 %l) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vlutvwh(<16 x i32> %a, <16 x i32> %b, i32 %l)
  store <32 x i32> %0, ptr @l, align 128
  ret void
}

; CHECK-LABEL: test56:
; CHECK: v{{[0-9]+}}.w = vinsert(r{{[0-9]+}})
define void @test56(i32 %b) #0 {
entry:
  %0 = load <16 x i32>, ptr @k, align 64
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vinsertwr(<16 x i32> %0, i32 %b)
  store <16 x i32> %1, ptr @k, align 64
  ret void
}

declare <32 x i32> @llvm.hexagon.V6.vrmpybusi(<32 x i32>, i32, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vrsadubi(<32 x i32>, i32, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vrmpyubi(<32 x i32>, i32, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32>, <32 x i32>, i32, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vrsadubi.acc(<32 x i32>, <32 x i32>, i32, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vrmpyubi.acc(<32 x i32>, <32 x i32>, i32, i32) #0
declare <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vlalignb(<16 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vasrwh(<16 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vasrwhsat(<16 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vasrwhrndsat(<16 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vasrwuhsat(<16 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vasrhubsat(<16 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vasrhubrndsat(<16 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vasrhbrndsat(<16 x i32>, <16 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vunpackob(<32 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vunpackoh(<32 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vmux(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.pred.and(<64 x i1>, <64 x i1>) #0
declare <64 x i1> @llvm.hexagon.V6.pred.or(<64 x i1>, <64 x i1>) #0
declare <64 x i1> @llvm.hexagon.V6.pred.not(<64 x i1>) #0
declare <64 x i1> @llvm.hexagon.V6.pred.xor(<64 x i1>, <64 x i1>) #0
declare <64 x i1> @llvm.hexagon.V6.pred.or.n(<64 x i1>, <64 x i1>) #0
declare <64 x i1> @llvm.hexagon.V6.pred.and.n(<64 x i1>, <64 x i1>) #0
declare <64 x i1> @llvm.hexagon.V6.vgtub(<16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.vgth(<16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.veqh(<16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.vgtw(<16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.veqw(<16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.vgtuh(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vandqrt.acc(<16 x i32>, <64 x i1>, i32) #0
declare <64 x i1> @llvm.hexagon.V6.vandvrt.acc(<64 x i1>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1>, i32) #0
declare <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32>, i32) #0
declare i64 @llvm.hexagon.S6.rol.i.p(i64, i32) #0
declare i64 @llvm.hexagon.S6.rol.i.p.acc(i64, i64, i32) #0
declare i64 @llvm.hexagon.S6.rol.i.p.and(i64, i64, i32) #0
declare i64 @llvm.hexagon.S6.rol.i.p.nac(i64, i64, i32) #0
declare i64 @llvm.hexagon.S6.rol.i.p.or(i64, i64, i32) #0
declare i64 @llvm.hexagon.S6.rol.i.p.xacc(i64, i64, i32) #0
declare i32 @llvm.hexagon.S6.rol.i.r(i32, i32) #0
declare i32 @llvm.hexagon.S6.rol.i.r.acc(i32, i32, i32) #0
declare i32 @llvm.hexagon.S6.rol.i.r.and(i32, i32, i32) #0
declare i32 @llvm.hexagon.S6.rol.i.r.nac(i32, i32, i32) #0
declare i32 @llvm.hexagon.S6.rol.i.r.or(i32, i32, i32) #0
declare i32 @llvm.hexagon.S6.rol.i.r.xacc(i32, i32, i32) #0
declare i32 @llvm.hexagon.V6.extractw(<16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.lvsplatw(i32) #0
declare <64 x i1> @llvm.hexagon.V6.pred.scalar2(i32) #0
declare <16 x i32> @llvm.hexagon.V6.vlutvvb(<16 x i32>, <16 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vlutvwh(<16 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vlutvvb.oracc(<16 x i32>, <16 x i32>, <16 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vlutvwh.oracc(<32 x i32>, <16 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vinsertwr(<16 x i32>, i32) #0

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
