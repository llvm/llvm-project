; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK-LABEL: test1:
; CHECK: v{{[0-9]+}} = vand(v{{[0-9]+}},v{{[0-9]+}})
define <16 x i32> @test1(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vand(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test2:
; CHECK: v{{[0-9]+}} = vor(v{{[0-9]+}},v{{[0-9]+}})
define <16 x i32> @test2(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vor(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test3:
; CHECK: v{{[0-9]+}} = vxor(v{{[0-9]+}},v{{[0-9]+}})
define <16 x i32> @test3(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vxor(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test4:
; CHECK: v{{[0-9]+}}.w = vadd(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
define <16 x i32> @test4(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test5:
; CHECK: v{{[0-9]+}}.ub = vadd(v{{[0-9]+}}.ub,v{{[0-9]+}}.ub):sat
define <16 x i32> @test5(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vaddubsat(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test6:
; CHECK: v{{[0-9]+}}.uh = vadd(v{{[0-9]+}}.uh,v{{[0-9]+}}.uh):sat
define <16 x i32> @test6(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vadduhsat(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test7:
; CHECK: v{{[0-9]+}}.h = vadd(v{{[0-9]+}}.h,v{{[0-9]+}}.h):sat
define <16 x i32> @test7(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vaddhsat(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test8:
; CHECK: v{{[0-9]+}}.w = vadd(v{{[0-9]+}}.w,v{{[0-9]+}}.w):sat
define <16 x i32> @test8(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vaddwsat(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test9:
; CHECK: v{{[0-9]+}}.b = vsub(v{{[0-9]+}}.b,v{{[0-9]+}}.b)
define <16 x i32> @test9(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vsubb(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test10:
; CHECK: v{{[0-9]+}}.h = vsub(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define <16 x i32> @test10(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vsubh(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test11:
; CHECK: v{{[0-9]+}}.w = vsub(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
define <16 x i32> @test11(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vsubw(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test12:
; CHECK: v{{[0-9]+}}.ub = vsub(v{{[0-9]+}}.ub,v{{[0-9]+}}.ub):sat
define <16 x i32> @test12(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vsububsat(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test13:
; CHECK: v{{[0-9]+}}.uh = vsub(v{{[0-9]+}}.uh,v{{[0-9]+}}.uh):sat
define <16 x i32> @test13(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vsubuhsat(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test14:
; CHECK: v{{[0-9]+}}.h = vsub(v{{[0-9]+}}.h,v{{[0-9]+}}.h):sat
define <16 x i32> @test14(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vsubhsat(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test15:
; CHECK: v{{[0-9]+}}.w = vsub(v{{[0-9]+}}.w,v{{[0-9]+}}.w):sat
define <16 x i32> @test15(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vsubwsat(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test16:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.b = vadd(v{{[0-9]+}}:{{[0-9]+}}.b,v{{[0-9]+}}:{{[0-9]+}}.b)
define <32 x i32> @test16(<32 x i32> %a, <32 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vaddb.dv(<32 x i32> %a, <32 x i32> %b)
  ret <32 x i32> %0
}

; CHECK-LABEL: test17:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h = vadd(v{{[0-9]+}}:{{[0-9]+}}.h,v{{[0-9]+}}:{{[0-9]+}}.h)
define <32 x i32> @test17(<32 x i32> %a, <32 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vaddh.dv(<32 x i32> %a, <32 x i32> %b)
  ret <32 x i32> %0
}

; CHECK-LABEL: test18:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w = vadd(v{{[0-9]+}}:{{[0-9]+}}.w,v{{[0-9]+}}:{{[0-9]+}}.w)
define <32 x i32> @test18(<32 x i32> %a, <32 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.dv(<32 x i32> %a, <32 x i32> %b)
  ret <32 x i32> %0
}

; CHECK-LABEL: test19:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.ub = vadd(v{{[0-9]+}}:{{[0-9]+}}.ub,v{{[0-9]+}}:{{[0-9]+}}.ub):sat
define <32 x i32> @test19(<32 x i32> %a, <32 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vaddubsat.dv(<32 x i32> %a, <32 x i32> %b)
  ret <32 x i32> %0
}

; CHECK-LABEL: test20:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.uh = vadd(v{{[0-9]+}}:{{[0-9]+}}.uh,v{{[0-9]+}}:{{[0-9]+}}.uh):sat
define <32 x i32> @test20(<32 x i32> %a, <32 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vadduhsat.dv(<32 x i32> %a, <32 x i32> %b)
  ret <32 x i32> %0
}

; CHECK-LABEL: test21:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h = vadd(v{{[0-9]+}}:{{[0-9]+}}.h,v{{[0-9]+}}:{{[0-9]+}}.h):sat
define <32 x i32> @test21(<32 x i32> %a, <32 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vaddhsat.dv(<32 x i32> %a, <32 x i32> %b)
  ret <32 x i32> %0
}

; CHECK-LABEL: test22:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w = vadd(v{{[0-9]+}}:{{[0-9]+}}.w,v{{[0-9]+}}:{{[0-9]+}}.w):sat
define <32 x i32> @test22(<32 x i32> %a, <32 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vaddwsat.dv(<32 x i32> %a, <32 x i32> %b)
  ret <32 x i32> %0
}

; CHECK-LABEL: test23:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.b = vsub(v{{[0-9]+}}:{{[0-9]+}}.b,v{{[0-9]+}}:{{[0-9]+}}.b)
define <32 x i32> @test23(<32 x i32> %a, <32 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vsubb.dv(<32 x i32> %a, <32 x i32> %b)
  ret <32 x i32> %0
}

; CHECK-LABEL: test24:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h = vsub(v{{[0-9]+}}:{{[0-9]+}}.h,v{{[0-9]+}}:{{[0-9]+}}.h)
define <32 x i32> @test24(<32 x i32> %a, <32 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vsubh.dv(<32 x i32> %a, <32 x i32> %b)
  ret <32 x i32> %0
}

; CHECK-LABEL: test25:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w = vsub(v{{[0-9]+}}:{{[0-9]+}}.w,v{{[0-9]+}}:{{[0-9]+}}.w)
define <32 x i32> @test25(<32 x i32> %a, <32 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vsubw.dv(<32 x i32> %a, <32 x i32> %b)
  ret <32 x i32> %0
}

; CHECK-LABEL: test26:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.ub = vsub(v{{[0-9]+}}:{{[0-9]+}}.ub,v{{[0-9]+}}:{{[0-9]+}}.ub):sat
define <32 x i32> @test26(<32 x i32> %a, <32 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vsububsat.dv(<32 x i32> %a, <32 x i32> %b)
  ret <32 x i32> %0
}

; CHECK-LABEL: test27:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.uh = vsub(v{{[0-9]+}}:{{[0-9]+}}.uh,v{{[0-9]+}}:{{[0-9]+}}.uh):sat
define <32 x i32> @test27(<32 x i32> %a, <32 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vsubuhsat.dv(<32 x i32> %a, <32 x i32> %b)
  ret <32 x i32> %0
}

; CHECK-LABEL: test28:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h = vsub(v{{[0-9]+}}:{{[0-9]+}}.h,v{{[0-9]+}}:{{[0-9]+}}.h):sat
define <32 x i32> @test28(<32 x i32> %a, <32 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vsubhsat.dv(<32 x i32> %a, <32 x i32> %b)
  ret <32 x i32> %0
}

; CHECK-LABEL: test29:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w = vsub(v{{[0-9]+}}:{{[0-9]+}}.w,v{{[0-9]+}}:{{[0-9]+}}.w):sat
define <32 x i32> @test29(<32 x i32> %a, <32 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vsubwsat.dv(<32 x i32> %a, <32 x i32> %b)
  ret <32 x i32> %0
}

; CHECK-LABEL: test30:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h = vadd(v{{[0-9]+}}.ub,v{{[0-9]+}}.ub)
define <32 x i32> @test30(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vaddubh(<16 x i32> %a, <16 x i32> %b)
  ret <32 x i32> %0
}

; CHECK-LABEL: test31:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w = vadd(v{{[0-9]+}}.uh,v{{[0-9]+}}.uh)
define <32 x i32> @test31(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vadduhw(<16 x i32> %a, <16 x i32> %b)
  ret <32 x i32> %0
}

; CHECK-LABEL: test32:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w = vadd(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define <32 x i32> @test32(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vaddhw(<16 x i32> %a, <16 x i32> %b)
  ret <32 x i32> %0
}

; CHECK-LABEL: test33:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h = vsub(v{{[0-9]+}}.ub,v{{[0-9]+}}.ub)
define <32 x i32> @test33(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vsububh(<16 x i32> %a, <16 x i32> %b)
  ret <32 x i32> %0
}

; CHECK-LABEL: test34:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w = vsub(v{{[0-9]+}}.uh,v{{[0-9]+}}.uh)
define <32 x i32> @test34(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vsubuhw(<16 x i32> %a, <16 x i32> %b)
  ret <32 x i32> %0
}

; CHECK-LABEL: test35:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w = vsub(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define <32 x i32> @test35(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vsubhw(<16 x i32> %a, <16 x i32> %b)
  ret <32 x i32> %0
}

; CHECK-LABEL: test36:
; CHECK: v{{[0-9]+}}.ub = vabsdiff(v{{[0-9]+}}.ub,v{{[0-9]+}}.ub)
define <16 x i32> @test36(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test37:
; CHECK: v{{[0-9]+}}.uh = vabsdiff(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define <16 x i32> @test37(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffh(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test38:
; CHECK: v{{[0-9]+}}.uh = vabsdiff(v{{[0-9]+}}.uh,v{{[0-9]+}}.uh)
define <16 x i32> @test38(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffuh(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test39:
; CHECK: v{{[0-9]+}}.uw = vabsdiff(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
define <16 x i32> @test39(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffw(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test40:
; CHECK: v{{[0-9]+}}.ub = vavg(v{{[0-9]+}}.ub,v{{[0-9]+}}.ub)
define <16 x i32> @test40(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vavgub(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test41:
; CHECK: v{{[0-9]+}}.uh = vavg(v{{[0-9]+}}.uh,v{{[0-9]+}}.uh)
define <16 x i32> @test41(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vavguh(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test42:
; CHECK: v{{[0-9]+}}.h = vavg(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define <16 x i32> @test42(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vavgh(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test43:
; CHECK: v{{[0-9]+}}.w = vavg(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
define <16 x i32> @test43(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vavgw(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test44:
; CHECK: v{{[0-9]+}}.b = vnavg(v{{[0-9]+}}.ub,v{{[0-9]+}}.ub)
define <16 x i32> @test44(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vnavgub(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test45:
; CHECK: v{{[0-9]+}}.h = vnavg(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define <16 x i32> @test45(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vnavgh(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test46:
; CHECK: v{{[0-9]+}}.w = vnavg(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
define <16 x i32> @test46(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vnavgw(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test47:
; CHECK: v{{[0-9]+}}.ub = vavg(v{{[0-9]+}}.ub,v{{[0-9]+}}.ub):rnd
define <16 x i32> @test47(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vavgubrnd(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test48:
; CHECK: v{{[0-9]+}}.uh = vavg(v{{[0-9]+}}.uh,v{{[0-9]+}}.uh):rnd
define <16 x i32> @test48(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vavguhrnd(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test49:
; CHECK: v{{[0-9]+}}.h = vavg(v{{[0-9]+}}.h,v{{[0-9]+}}.h):rnd
define <16 x i32> @test49(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vavghrnd(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test50:
; CHECK: v{{[0-9]+}}.w = vavg(v{{[0-9]+}}.w,v{{[0-9]+}}.w):rnd
define <16 x i32> @test50(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vavgwrnd(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test51:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h = vmpa(v{{[0-9]+}}:{{[0-9]+}}.ub,v{{[0-9]+}}:{{[0-9]+}}.ub)
define <32 x i32> @test51(<32 x i32> %a, <32 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vmpabuuv(<32 x i32> %a, <32 x i32> %b)
  ret <32 x i32> %0
}

; CHECK-LABEL: test52:
; CHECK: v{{[0-9]+}}.ub = vmin(v{{[0-9]+}}.ub,v{{[0-9]+}}.ub)
define <16 x i32> @test52(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vminub(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test53:
; CHECK: v{{[0-9]+}}.uh = vmin(v{{[0-9]+}}.uh,v{{[0-9]+}}.uh)
define <16 x i32> @test53(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vminuh(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test54:
; CHECK: v{{[0-9]+}}.h = vmin(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define <16 x i32> @test54(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vminh(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test55:
; CHECK: v{{[0-9]+}}.w = vmin(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
define <16 x i32> @test55(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vminw(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test56:
; CHECK: v{{[0-9]+}}.ub = vmax(v{{[0-9]+}}.ub,v{{[0-9]+}}.ub)
define <16 x i32> @test56(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vmaxub(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test57:
; CHECK: v{{[0-9]+}}.uh = vmax(v{{[0-9]+}}.uh,v{{[0-9]+}}.uh)
define <16 x i32> @test57(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vmaxuh(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test58:
; CHECK: v{{[0-9]+}}.h = vmax(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define <16 x i32> @test58(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vmaxh(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test59:
; CHECK: v{{[0-9]+}}.w = vmax(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
define <16 x i32> @test59(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vmaxw(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test60:
; CHECK: v{{[0-9]+}} = vdelta(v{{[0-9]+}},v{{[0-9]+}})
define <16 x i32> @test60(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vdelta(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test61:
; CHECK: v{{[0-9]+}} = vrdelta(v{{[0-9]+}},v{{[0-9]+}})
define <16 x i32> @test61(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vrdelta(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test62:
; CHECK: v{{[0-9]+}}.b = vdeale(v{{[0-9]+}}.b,v{{[0-9]+}}.b)
define <16 x i32> @test62(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vdealb4w(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test63:
; CHECK: v{{[0-9]+}}.b = vshuffe(v{{[0-9]+}}.b,v{{[0-9]+}}.b)
define <16 x i32> @test63(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vshuffeb(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test64:
; CHECK: v{{[0-9]+}}.b = vshuffo(v{{[0-9]+}}.b,v{{[0-9]+}}.b)
define <16 x i32> @test64(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vshuffob(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test65:
; CHECK: v{{[0-9]+}}.h = vshuffe(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define <16 x i32> @test65(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vshufeh(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test66:
; CHECK: v{{[0-9]+}}.h = vshuffo(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define <16 x i32> @test66(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vshufoh(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test67:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h = vshuffoe(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define <32 x i32> @test67(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vshufoeh(<16 x i32> %a, <16 x i32> %b)
  ret <32 x i32> %0
}

; CHECK-LABEL: test68:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.b = vshuffoe(v{{[0-9]+}}.b,v{{[0-9]+}}.b)
define <32 x i32> @test68(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vshufoeb(<16 x i32> %a, <16 x i32> %b)
  ret <32 x i32> %0
}

; CHECK-LABEL: test69:
; CHECK: v{{[0-9]+}}:{{[0-9]+}} = vcombine(v{{[0-9]+}},v{{[0-9]+}})
define <32 x i32> @test69(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %a, <16 x i32> %b)
  ret <32 x i32> %0
}

; CHECK-LABEL: test70:
; CHECK: v{{[0-9]+}}.ub = vsat(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define <16 x i32> @test70(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test71:
; CHECK: v{{[0-9]+}}.h = vsat(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
define <16 x i32> @test71(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vsatwh(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test72:
; CHECK: v{{[0-9]+}}.h = vround(v{{[0-9]+}}.w,v{{[0-9]+}}.w):sat
define <16 x i32> @test72(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vroundwh(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test73:
; CHECK: v{{[0-9]+}}.uh = vround(v{{[0-9]+}}.w,v{{[0-9]+}}.w):sat
define <16 x i32> @test73(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vroundwuh(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test74:
; CHECK: v{{[0-9]+}}.b = vround(v{{[0-9]+}}.h,v{{[0-9]+}}.h):sat
define <16 x i32> @test74(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vroundhb(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test75:
; CHECK: v{{[0-9]+}}.ub = vround(v{{[0-9]+}}.h,v{{[0-9]+}}.h):sat
define <16 x i32> @test75(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vroundhub(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test76:
; CHECK: v{{[0-9]+}}.w = vasr(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
define <16 x i32> @test76(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vasrwv(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test77:
; CHECK: v{{[0-9]+}}.w = vlsr(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
define <16 x i32> @test77(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vlsrwv(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test78:
; CHECK: v{{[0-9]+}}.h = vlsr(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define <16 x i32> @test78(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vlsrhv(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test79:
; CHECK: v{{[0-9]+}}.h = vasr(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define <16 x i32> @test79(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vasrhv(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test80:
; CHECK: v{{[0-9]+}}.w = vasl(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
define <16 x i32> @test80(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vaslwv(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test81:
; CHECK: v{{[0-9]+}}.h = vasl(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define <16 x i32> @test81(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vaslhv(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test82:
; CHECK: v{{[0-9]+}}.b = vadd(v{{[0-9]+}}.b,v{{[0-9]+}}.b)
define <16 x i32> @test82(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vaddb(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test83:
; CHECK: v{{[0-9]+}}.h = vadd(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define <16 x i32> @test83(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %0
}

; CHECK-LABEL: test84:
; CHECK: if (q{{[0-3]}}) v{{[0-9]+}}.b += v{{[0-9]+}}.b
define <16 x i32> @test84(<16 x i32> %a, <16 x i32> %b, <16 x i32> %c) #0 {
entry:
  %0 = bitcast <16 x i32> %a to <512 x i1>
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vaddbq(<512 x i1> %0, <16 x i32> %c, <16 x i32> %b)
  ret <16 x i32> %1
}

; CHECK-LABEL: test85:
; CHECK: if (q{{[0-3]}}) v{{[0-9]+}}.h += v{{[0-9]+}}.h
define <16 x i32> @test85(<16 x i32> %a, <16 x i32> %b, <16 x i32> %c) #0 {
entry:
  %0 = bitcast <16 x i32> %a to <512 x i1>
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vaddhq(<512 x i1> %0, <16 x i32> %c, <16 x i32> %b)
  ret <16 x i32> %1
}

; CHECK-LABEL: test86:
; CHECK: if (q{{[0-3]}}) v{{[0-9]+}}.w += v{{[0-9]+}}.w
define <16 x i32> @test86(<16 x i32> %a, <16 x i32> %b, <16 x i32> %c) #0 {
entry:
  %0 = bitcast <16 x i32> %a to <512 x i1>
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vaddwq(<512 x i1> %0, <16 x i32> %c, <16 x i32> %b)
  ret <16 x i32> %1
}

; CHECK-LABEL: test87:
; CHECK: if (!q{{[0-3]}}) v{{[0-9]+}}.b += v{{[0-9]+}}.b
define <16 x i32> @test87(<16 x i32> %a, <16 x i32> %b, <16 x i32> %c) #0 {
entry:
  %0 = bitcast <16 x i32> %a to <512 x i1>
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<512 x i1> %0, <16 x i32> %c, <16 x i32> %b)
  ret <16 x i32> %1
}

; CHECK-LABEL: test88:
; CHECK: if (!q{{[0-3]}}) v{{[0-9]+}}.h += v{{[0-9]+}}.h
define <16 x i32> @test88(<16 x i32> %a, <16 x i32> %b, <16 x i32> %c) #0 {
entry:
  %0 = bitcast <16 x i32> %a to <512 x i1>
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vaddhnq(<512 x i1> %0, <16 x i32> %c, <16 x i32> %b)
  ret <16 x i32> %1
}

; CHECK-LABEL: test89:
; CHECK: if (!q{{[0-3]}}) v{{[0-9]+}}.w += v{{[0-9]+}}.w
define <16 x i32> @test89(<16 x i32> %a, <16 x i32> %b, <16 x i32> %c) #0 {
entry:
  %0 = bitcast <16 x i32> %a to <512 x i1>
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vaddwnq(<512 x i1> %0, <16 x i32> %c, <16 x i32> %b)
  ret <16 x i32> %1
}

; CHECK-LABEL: test90:
; CHECK: if (q{{[0-3]}}) v{{[0-9]+}}.b -= v{{[0-9]+}}.b
define <16 x i32> @test90(<16 x i32> %a, <16 x i32> %b, <16 x i32> %c) #0 {
entry:
  %0 = bitcast <16 x i32> %a to <512 x i1>
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vsubbq(<512 x i1> %0, <16 x i32> %c, <16 x i32> %b)
  ret <16 x i32> %1
}

; CHECK-LABEL: test91:
; CHECK: if (q{{[0-3]}}) v{{[0-9]+}}.h -= v{{[0-9]+}}.h
define <16 x i32> @test91(<16 x i32> %a, <16 x i32> %b, <16 x i32> %c) #0 {
entry:
  %0 = bitcast <16 x i32> %a to <512 x i1>
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vsubhq(<512 x i1> %0, <16 x i32> %c, <16 x i32> %b)
  ret <16 x i32> %1
}

; CHECK-LABEL: test92:
; CHECK: if (q{{[0-3]}}) v{{[0-9]+}}.w -= v{{[0-9]+}}.w
define <16 x i32> @test92(<16 x i32> %a, <16 x i32> %b, <16 x i32> %c) #0 {
entry:
  %0 = bitcast <16 x i32> %a to <512 x i1>
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vsubwq(<512 x i1> %0, <16 x i32> %c, <16 x i32> %b)
  ret <16 x i32> %1
}

; CHECK-LABEL: test93:
; CHECK: if (!q{{[0-3]}}) v{{[0-9]+}}.b -= v{{[0-9]+}}.b
define <16 x i32> @test93(<16 x i32> %a, <16 x i32> %b, <16 x i32> %c) #0 {
entry:
  %0 = bitcast <16 x i32> %a to <512 x i1>
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vsubbnq(<512 x i1> %0, <16 x i32> %c, <16 x i32> %b)
  ret <16 x i32> %1
}

; CHECK-LABEL: test94:
; CHECK: if (!q{{[0-3]}}) v{{[0-9]+}}.h -= v{{[0-9]+}}.h
define <16 x i32> @test94(<16 x i32> %a, <16 x i32> %b, <16 x i32> %c) #0 {
entry:
  %0 = bitcast <16 x i32> %a to <512 x i1>
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vsubhnq(<512 x i1> %0, <16 x i32> %c, <16 x i32> %b)
  ret <16 x i32> %1
}

; CHECK-LABEL: test95:
; CHECK: if (!q{{[0-3]}}) v{{[0-9]+}}.w -= v{{[0-9]+}}.w
define <16 x i32> @test95(<16 x i32> %a, <16 x i32> %b, <16 x i32> %c) #0 {
entry:
  %0 = bitcast <16 x i32> %a to <512 x i1>
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vsubwnq(<512 x i1> %0, <16 x i32> %c, <16 x i32> %b)
  ret <16 x i32> %1
}

; CHECK-LABEL: test96:
; CHECK: v{{[0-9]+}}.h = vabs(v{{[0-9]+}}.h)
define <16 x i32> @test96(<16 x i32> %a) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vabsh(<16 x i32> %a)
  ret <16 x i32> %0
}

; CHECK-LABEL: test97:
; CHECK: v{{[0-9]+}}.h = vabs(v{{[0-9]+}}.h):sat
define <16 x i32> @test97(<16 x i32> %a) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vabsh.sat(<16 x i32> %a)
  ret <16 x i32> %0
}

; CHECK-LABEL: test98:
; CHECK: v{{[0-9]+}}.w = vabs(v{{[0-9]+}}.w)
define <16 x i32> @test98(<16 x i32> %a) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vabsw(<16 x i32> %a)
  ret <16 x i32> %0
}

; CHECK-LABEL: test99:
; CHECK: v{{[0-9]+}}.w = vabs(v{{[0-9]+}}.w):sat
define <16 x i32> @test99(<16 x i32> %a) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vabsw.sat(<16 x i32> %a)
  ret <16 x i32> %0
}

; CHECK-LABEL: test100:
; CHECK: v{{[0-9]+}} = vnot(v{{[0-9]+}})
define <16 x i32> @test100(<16 x i32> %a) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vnot(<16 x i32> %a)
  ret <16 x i32> %0
}

; CHECK-LABEL: test101:
; CHECK: v{{[0-9]+}}.h = vdeal(v{{[0-9]+}}.h)
define <16 x i32> @test101(<16 x i32> %a) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vdealh(<16 x i32> %a)
  ret <16 x i32> %0
}

; CHECK-LABEL: test102:
; CHECK: v{{[0-9]+}}.b = vdeal(v{{[0-9]+}}.b)
define <16 x i32> @test102(<16 x i32> %a) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vdealb(<16 x i32> %a)
  ret <16 x i32> %0
}

; CHECK-LABEL: test103:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.uh = vunpack(v{{[0-9]+}}.ub)
define <32 x i32> @test103(<16 x i32> %a) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vunpackub(<16 x i32> %a)
  ret <32 x i32> %0
}

; CHECK-LABEL: test104:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.uw = vunpack(v{{[0-9]+}}.uh)
define <32 x i32> @test104(<16 x i32> %a) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vunpackuh(<16 x i32> %a)
  ret <32 x i32> %0
}

; CHECK-LABEL: test105:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h = vunpack(v{{[0-9]+}}.b)
define <32 x i32> @test105(<16 x i32> %a) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vunpackb(<16 x i32> %a)
  ret <32 x i32> %0
}

; CHECK-LABEL: test106:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w = vunpack(v{{[0-9]+}}.h)
define <32 x i32> @test106(<16 x i32> %a) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vunpackh(<16 x i32> %a)
  ret <32 x i32> %0
}

; CHECK-LABEL: test107:
; CHECK: v{{[0-9]+}}.h = vshuff(v{{[0-9]+}}.h)
define <16 x i32> @test107(<16 x i32> %a) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vshuffh(<16 x i32> %a)
  ret <16 x i32> %0
}

; CHECK-LABEL: test108:
; CHECK: v{{[0-9]+}}.b = vshuff(v{{[0-9]+}}.b)
define <16 x i32> @test108(<16 x i32> %a) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vshuffb(<16 x i32> %a)
  ret <16 x i32> %0
}

; CHECK-LABEL: test109:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.uh = vzxt(v{{[0-9]+}}.ub)
define <32 x i32> @test109(<16 x i32> %a) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vzb(<16 x i32> %a)
  ret <32 x i32> %0
}

; CHECK-LABEL: test110:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.uw = vzxt(v{{[0-9]+}}.uh)
define <32 x i32> @test110(<16 x i32> %a) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vzh(<16 x i32> %a)
  ret <32 x i32> %0
}

; CHECK-LABEL: test111:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h = vsxt(v{{[0-9]+}}.b)
define <32 x i32> @test111(<16 x i32> %a) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vsb(<16 x i32> %a)
  ret <32 x i32> %0
}

; CHECK-LABEL: test112:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w = vsxt(v{{[0-9]+}}.h)
define <32 x i32> @test112(<16 x i32> %a) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vsh(<16 x i32> %a)
  ret <32 x i32> %0
}

; CHECK-LABEL: test113:
; CHECK: v{{[0-9]+}} = v{{[0-9]+}}
define <16 x i32> @test113(<16 x i32> %a) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vassign(<16 x i32> %a)
  ret <16 x i32> %0
}

declare <16 x i32> @llvm.hexagon.V6.vadduhsat(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vaddhsat(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vaddwsat(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vsubb(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vsubh(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vsubw(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vsububsat(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vsubuhsat(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vsubhsat(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vsubwsat(<16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vaddb.dv(<32 x i32>, <32 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vaddh.dv(<32 x i32>, <32 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vaddw.dv(<32 x i32>, <32 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vaddubsat.dv(<32 x i32>, <32 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vadduhsat.dv(<32 x i32>, <32 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vaddhsat.dv(<32 x i32>, <32 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vaddwsat.dv(<32 x i32>, <32 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vsubb.dv(<32 x i32>, <32 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vsubh.dv(<32 x i32>, <32 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vsubw.dv(<32 x i32>, <32 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vsububsat.dv(<32 x i32>, <32 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vsubuhsat.dv(<32 x i32>, <32 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vsubhsat.dv(<32 x i32>, <32 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vsubwsat.dv(<32 x i32>, <32 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vaddubh(<16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vadduhw(<16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vaddhw(<16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vsububh(<16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vsubuhw(<16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vsubhw(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vabsdiffh(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vabsdiffuh(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vabsdiffw(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vavgub(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vavguh(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vavgh(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vavgw(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vnavgub(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vnavgh(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vnavgw(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vavgubrnd(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vavghrnd(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vavguhrnd(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vavgwrnd(<16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vmpabuuv(<32 x i32>, <32 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vand(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vminub(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vminuh(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vminh(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vminw(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vmaxub(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vmaxuh(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vmaxh(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vmaxw(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vdelta(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vrdelta(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vdealb4w(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vshuffob(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vshuffeb(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vshufeh(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vshufoh(<16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vshufoeh(<16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vshufoeb(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vsatwh(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vroundwh(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vroundhb(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vroundwuh(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vroundhub(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vasrwv(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vlsrwv(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vasrhv(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vlsrhv(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vaslwv(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vaslhv(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vaddb(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vor(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vxor(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vaddubsat(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vaddbq(<512 x i1>, <16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vaddhq(<512 x i1>, <16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vaddwq(<512 x i1>, <16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vaddbnq(<512 x i1>, <16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vaddhnq(<512 x i1>, <16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vaddwnq(<512 x i1>, <16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vsubbq(<512 x i1>, <16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vsubhq(<512 x i1>, <16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vsubwq(<512 x i1>, <16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vsubbnq(<512 x i1>, <16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vsubhnq(<512 x i1>, <16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vsubwnq(<512 x i1>, <16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vabsh(<16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vabsh.sat(<16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vabsw(<16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vabsw.sat(<16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vnot(<16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vdealh(<16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vdealb(<16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vunpackub(<16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vunpackuh(<16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vunpackb(<16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vunpackh(<16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vshuffh(<16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vshuffb(<16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vzb(<16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vzh(<16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vsb(<16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vsh(<16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vassign(<16 x i32>) #0

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }

