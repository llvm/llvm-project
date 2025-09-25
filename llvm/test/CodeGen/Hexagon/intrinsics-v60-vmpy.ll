; RUN: llc -mtriple=hexagon < %s | FileCheck %s

@c = external global <32 x i32>
@d = external global <16 x i32>

; CHECK-LABEL: test1:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h = vtmpy(v{{[0-9]+}}:{{[0-9]+}}.b,r{{[0-9]+}}.b)
define void @test1(<32 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vtmpyb(<32 x i32> %a, i32 %b)
  store <32 x i32> %0, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test2:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h = vtmpy(v{{[0-9]+}}:{{[0-9]+}}.ub,r{{[0-9]+}}.b)
define void @test2(<32 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vtmpybus(<32 x i32> %a, i32 %b)
  store <32 x i32> %0, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test3:
; CHECK: v{{[0-9]+}}.w = vdmpy(v{{[0-9]+}}.h,r{{[0-9]+}}.b)
define void @test3(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vdmpyhb(<16 x i32> %a, i32 %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test4:
; CHECK: v{{[0-9]+}}.uw = vrmpy(v{{[0-9]+}}.ub,r{{[0-9]+}}.ub)
define void @test4(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vrmpyub(<16 x i32> %a, i32 %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test5:
; CHECK: v{{[0-9]+}}.w = vrmpy(v{{[0-9]+}}.ub,r{{[0-9]+}}.b)
define void @test5(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vrmpybus(<16 x i32> %a, i32 %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test6:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.uw = vdsad(v{{[0-9]+}}:{{[0-9]+}}.uh,r{{[0-9]+}}.uh)
define void @test6(<32 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vdsaduh(<32 x i32> %a, i32 %b)
  store <32 x i32> %0, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test7:
; CHECK: v{{[0-9]+}}.h = vdmpy(v{{[0-9]+}}.ub,r{{[0-9]+}}.b)
define void @test7(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vdmpybus(<16 x i32> %a, i32 %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test8:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h = vdmpy(v{{[0-9]+}}:{{[0-9]+}}.ub,r{{[0-9]+}}.b)
define void @test8(<32 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv(<32 x i32> %a, i32 %b)
  store <32 x i32> %0, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test9:
; CHECK: v{{[0-9]+}}.w = vdmpy(v{{[0-9]+}}.h,r{{[0-9]+}}.uh):sat
define void @test9(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vdmpyhsusat(<16 x i32> %a, i32 %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test10:
; CHECK: v{{[0-9]+}}.w = vdmpy(v{{[0-9]+}}:{{[0-9]+}}.h,r{{[0-9]+}}.uh,#1):sat
define void @test10(<32 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vdmpyhsuisat(<32 x i32> %a, i32 %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test11:
; CHECK: v{{[0-9]+}}.w = vdmpy(v{{[0-9]+}}.h,r{{[0-9]+}}.h):sat
define void @test11(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vdmpyhsat(<16 x i32> %a, i32 %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test12:
; CHECK: v{{[0-9]+}}.w = vdmpy(v{{[0-9]+}}:{{[0-9]+}}.h,r{{[0-9]+}}.h):sat
define void @test12(<32 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vdmpyhisat(<32 x i32> %a, i32 %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test13:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w = vdmpy(v{{[0-9]+}}:{{[0-9]+}}.h,r{{[0-9]+}}.b)
define void @test13(<32 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vdmpyhb.dv(<32 x i32> %a, i32 %b)
  store <32 x i32> %0, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test14:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h = vmpy(v{{[0-9]+}}.ub,r{{[0-9]+}}.b)
define void @test14(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vmpybus(<16 x i32> %a, i32 %b)
  store <32 x i32> %0, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test15:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h = vmpa(v{{[0-9]+}}:{{[0-9]+}}.ub,r{{[0-9]+}}.b)
define void @test15(<32 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus(<32 x i32> %a, i32 %b)
  store <32 x i32> %0, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test16:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w = vmpa(v{{[0-9]+}}:{{[0-9]+}}.h,r{{[0-9]+}}.b)
define void @test16(<32 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vmpahb(<32 x i32> %a, i32 %b)
  store <32 x i32> %0, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test17:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w = vmpy(v{{[0-9]+}}.h,r{{[0-9]+}}.h)
define void @test17(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vmpyh(<16 x i32> %a, i32 %b)
  store <32 x i32> %0, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test18:
; CHECK: v{{[0-9]+}}.h = vmpy(v{{[0-9]+}}.h,r{{[0-9]+}}.h):<<1:sat
define void @test18(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vmpyhss(<16 x i32> %a, i32 %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test19:
; CHECK: v{{[0-9]+}}.h = vmpy(v{{[0-9]+}}.h,r{{[0-9]+}}.h):<<1:rnd:sat
define void @test19(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vmpyhsrs(<16 x i32> %a, i32 %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test20:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.uw = vmpy(v{{[0-9]+}}.uh,r{{[0-9]+}}.uh)
define void @test20(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vmpyuh(<16 x i32> %a, i32 %b)
  store <32 x i32> %0, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test21:
; CHECK: v{{[0-9]+}}.h = vmpyi(v{{[0-9]+}}.h,r{{[0-9]+}}.b)
define void @test21(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vmpyihb(<16 x i32> %a, i32 %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test22:
; CHECK: v{{[0-9]+}} = vror(v{{[0-9]+}},r{{[0-9]+}})
define void @test22(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vror(<16 x i32> %a, i32 %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test23:
; CHECK: v{{[0-9]+}}.w = vasr(v{{[0-9]+}}.w,r{{[0-9]+}})
define void @test23(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vasrw(<16 x i32> %a, i32 %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test24:
; CHECK: v{{[0-9]+}}.h = vasr(v{{[0-9]+}}.h,r{{[0-9]+}})
define void @test24(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vasrh(<16 x i32> %a, i32 %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test25:
; CHECK: v{{[0-9]+}}.w = vasl(v{{[0-9]+}}.w,r{{[0-9]+}})
define void @test25(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vaslw(<16 x i32> %a, i32 %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test26:
; CHECK: v{{[0-9]+}}.h = vasl(v{{[0-9]+}}.h,r{{[0-9]+}})
define void @test26(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vaslh(<16 x i32> %a, i32 %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test27:
; CHECK: v{{[0-9]+}}.uw = vlsr(v{{[0-9]+}}.uw,r{{[0-9]+}})
define void @test27(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vlsrw(<16 x i32> %a, i32 %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test28:
; CHECK: v{{[0-9]+}}.uh = vlsr(v{{[0-9]+}}.uh,r{{[0-9]+}})
define void @test28(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vlsrh(<16 x i32> %a, i32 %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test29:
; CHECK: v{{[0-9]+}}.w = vmpyi(v{{[0-9]+}}.w,r{{[0-9]+}}.h)
define void @test29(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwh(<16 x i32> %a, i32 %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test30:
; CHECK: v{{[0-9]+}}.w = vmpyi(v{{[0-9]+}}.w,r{{[0-9]+}}.b)
define void @test30(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb(<16 x i32> %a, i32 %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test31:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w = vtmpy(v{{[0-9]+}}:{{[0-9]+}}.h,r{{[0-9]+}}.b)
define void @test31(<32 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vtmpyhb(<32 x i32> %a, i32 %b)
  store <32 x i32> %0, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test32:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.uh = vmpy(v{{[0-9]+}}.ub,r{{[0-9]+}}.ub)
define void @test32(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vmpyub(<16 x i32> %a, i32 %b)
  store <32 x i32> %0, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test33:
; CHECK: v{{[0-9]+}}.uw = vrmpy(v{{[0-9]+}}.ub,v{{[0-9]+}}.ub)
define void @test33(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vrmpyubv(<16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test34:
; CHECK: v{{[0-9]+}}.w = vrmpy(v{{[0-9]+}}.b,v{{[0-9]+}}.b)
define void @test34(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vrmpybv(<16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test35:
; CHECK: v{{[0-9]+}}.w = vrmpy(v{{[0-9]+}}.ub,v{{[0-9]+}}.b)
define void @test35(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vrmpybusv(<16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test36:
; CHECK: v{{[0-9]+}}.w = vdmpy(v{{[0-9]+}}.h,v{{[0-9]+}}.h):sat
define void @test36(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vdmpyhvsat(<16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test37:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h = vmpy(v{{[0-9]+}}.b,v{{[0-9]+}}.b)
define void @test37(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vmpybv(<16 x i32> %a, <16 x i32> %b)
  store <32 x i32> %0, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test38:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.uh = vmpy(v{{[0-9]+}}.ub,v{{[0-9]+}}.ub)
define void @test38(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vmpyubv(<16 x i32> %a, <16 x i32> %b)
  store <32 x i32> %0, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test39:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h = vmpy(v{{[0-9]+}}.ub,v{{[0-9]+}}.b)
define void @test39(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vmpybusv(<16 x i32> %a, <16 x i32> %b)
  store <32 x i32> %0, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test40:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w = vmpy(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define void @test40(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vmpyhv(<16 x i32> %a, <16 x i32> %b)
  store <32 x i32> %0, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test41:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.uw = vmpy(v{{[0-9]+}}.uh,v{{[0-9]+}}.uh)
define void @test41(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vmpyuhv(<16 x i32> %a, <16 x i32> %b)
  store <32 x i32> %0, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test42:
; CHECK: v{{[0-9]+}}.h = vmpy(v{{[0-9]+}}.h,v{{[0-9]+}}.h):<<1:rnd:sat
define void @test42(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vmpyhvsrs(<16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test43:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w = vmpy(v{{[0-9]+}}.h,v{{[0-9]+}}.uh)
define void @test43(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vmpyhus(<16 x i32> %a, <16 x i32> %b)
  store <32 x i32> %0, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test44:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h = vmpa(v{{[0-9]+}}:{{[0-9]+}}.ub,v{{[0-9]+}}:{{[0-9]+}}.b)
define void @test44(<32 x i32> %a, <32 x i32> %b) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vmpabusv(<32 x i32> %a, <32 x i32> %b)
  store <32 x i32> %0, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test45:
; CHECK: v{{[0-9]+}}.h = vmpyi(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define void @test45(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vmpyih(<16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test46:
; CHECK: v{{[0-9]+}}.w = vmpye(v{{[0-9]+}}.w,v{{[0-9]+}}.uh)
define void @test46(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vmpyewuh(<16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test47:
; CHECK: v{{[0-9]+}}.w = vmpyo(v{{[0-9]+}}.w,v{{[0-9]+}}.h):<<1:sat
define void @test47(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vmpyowh(<16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test48:
; CHECK: v{{[0-9]+}}.w = vmpyie(v{{[0-9]+}}.w,v{{[0-9]+}}.uh)
define void @test48(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiewuh(<16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test49:
; CHECK: v{{[0-9]+}}.w = vmpyio(v{{[0-9]+}}.w,v{{[0-9]+}}.h)
define void @test49(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiowh(<16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test50:
; CHECK: v{{[0-9]+}}.w = vmpyo(v{{[0-9]+}}.w,v{{[0-9]+}}.h):<<1:rnd:sat
define void @test50(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vmpyowh.rnd(<16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test51:
; CHECK: v{{[0-9]+}}.w = vmpyieo(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define void @test51(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.vmpyieoh(<16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %0, ptr @d, align 64
  ret void
}

declare <32 x i32> @llvm.hexagon.V6.vtmpyb(<32 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vtmpybus(<32 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vdmpyhb(<16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vrmpyub(<16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vrmpybus(<16 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vdsaduh(<32 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vdmpybus(<16 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vdmpybus.dv(<32 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vdmpyhsusat(<16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vdmpyhsuisat(<32 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vdmpyhsat(<16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vdmpyhisat(<32 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vdmpyhb.dv(<32 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vmpybus(<16 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vmpabus(<32 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vmpahb(<32 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vmpyh(<16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vmpyhss(<16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vmpyhsrs(<16 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vmpyuh(<16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vmpyihb(<16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vror(<16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vasrw(<16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vasrh(<16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vaslw(<16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vaslh(<16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vlsrw(<16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vlsrh(<16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vmpyiwh(<16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vmpyiwb(<16 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vtmpyhb(<32 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vmpyub(<16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vrmpyubv(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vrmpybv(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vrmpybusv(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vdmpyhvsat(<16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vmpybv(<16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vmpyubv(<16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vmpybusv(<16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vmpyhv(<16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vmpyuhv(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vmpyhvsrs(<16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vmpyhus(<16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vmpabusv(<32 x i32>, <32 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vmpyih(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vmpyewuh(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vmpyowh(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vmpyiewuh(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vmpyiowh(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vmpyowh.rnd(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vmpyieoh(<16 x i32>, <16 x i32>) #0

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
