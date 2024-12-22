; RUN: llc -march=hexagon < %s | FileCheck %s

@c = external global <32 x i32>
@d = external global <16 x i32>

; CHECK-LABEL: test1:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h += vtmpy(v{{[0-9]+}}:{{[0-9]+}}.b,r{{[0-9]+}}.b)
define void @test1(<32 x i32> %a, i32 %b) #0 {
entry:
  %0 = load <32 x i32>, ptr @c, align 128
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vtmpyb.acc(<32 x i32> %0, <32 x i32> %a, i32 %b)
  store <32 x i32> %1, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test2:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h += vtmpy(v{{[0-9]+}}:{{[0-9]+}}.ub,r{{[0-9]+}}.b)
define void @test2(<32 x i32> %a, i32 %b) #0 {
entry:
  %0 = load <32 x i32>, ptr @c, align 128
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vtmpybus.acc(<32 x i32> %0, <32 x i32> %a, i32 %b)
  store <32 x i32> %1, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test3:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w += vtmpy(v{{[0-9]+}}:{{[0-9]+}}.h,r{{[0-9]+}}.b)
define void @test3(<32 x i32> %a, i32 %b) #0 {
entry:
  %0 = load <32 x i32>, ptr @c, align 128
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vtmpyhb.acc(<32 x i32> %0, <32 x i32> %a, i32 %b)
  store <32 x i32> %1, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test4:
; CHECK: v{{[0-9]+}}.w += vdmpy(v{{[0-9]+}}.h,r{{[0-9]+}}.b)
define void @test4(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = load <16 x i32>, ptr @d, align 64
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vdmpyhb.acc(<16 x i32> %0, <16 x i32> %a, i32 %b)
  store <16 x i32> %1, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test5:
; CHECK: v{{[0-9]+}}.uw += vrmpy(v{{[0-9]+}}.ub,r{{[0-9]+}}.ub)
define void @test5(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = load <16 x i32>, ptr @d, align 64
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vrmpyub.acc(<16 x i32> %0, <16 x i32> %a, i32 %b)
  store <16 x i32> %1, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test6:
; CHECK: v{{[0-9]+}}.w += vrmpy(v{{[0-9]+}}.ub,r{{[0-9]+}}.b)
define void @test6(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = load <16 x i32>, ptr @d, align 64
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vrmpybus.acc(<16 x i32> %0, <16 x i32> %a, i32 %b)
  store <16 x i32> %1, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test7:
; CHECK: v{{[0-9]+}}.h += vdmpy(v{{[0-9]+}}.ub,r{{[0-9]+}}.b)
define void @test7(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = load <16 x i32>, ptr @d, align 64
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vdmpybus.acc(<16 x i32> %0, <16 x i32> %a, i32 %b)
  store <16 x i32> %1, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test8:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h += vdmpy(v{{[0-9]+}}:{{[0-9]+}}.ub,r{{[0-9]+}}.b)
define void @test8(<32 x i32> %a, i32 %b) #0 {
entry:
  %0 = load <32 x i32>, ptr @c, align 128
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32> %0, <32 x i32> %a, i32 %b)
  store <32 x i32> %1, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test9:
; CHECK: v{{[0-9]+}}.w += vdmpy(v{{[0-9]+}}.h,r{{[0-9]+}}.uh):sat
define void @test9(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = load <16 x i32>, ptr @d, align 64
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vdmpyhsusat.acc(<16 x i32> %0, <16 x i32> %a, i32 %b)
  store <16 x i32> %1, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test10:
; CHECK: v{{[0-9]+}}.w += vdmpy(v{{[0-9]+}}:{{[0-9]+}}.h,r{{[0-9]+}}.uh,#1):sat
define void @test10(<32 x i32> %a, i32 %b) #0 {
entry:
  %0 = load <16 x i32>, ptr @d, align 64
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vdmpyhsuisat.acc(<16 x i32> %0, <32 x i32> %a, i32 %b)
  store <16 x i32> %1, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test11:
; CHECK: v{{[0-9]+}}.w += vdmpy(v{{[0-9]+}}:{{[0-9]+}}.h,r{{[0-9]+}}.h):sat
define void @test11(<32 x i32> %a, i32 %b) #0 {
entry:
  %0 = load <16 x i32>, ptr @d, align 64
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vdmpyhisat.acc(<16 x i32> %0, <32 x i32> %a, i32 %b)
  store <16 x i32> %1, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test12:
; CHECK: v{{[0-9]+}}.w += vdmpy(v{{[0-9]+}}.h,r{{[0-9]+}}.h):sat
define void @test12(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = load <16 x i32>, ptr @d, align 64
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vdmpyhsat.acc(<16 x i32> %0, <16 x i32> %a, i32 %b)
  store <16 x i32> %1, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test13:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w += vdmpy(v{{[0-9]+}}:{{[0-9]+}}.h,r{{[0-9]+}}.b)
define void @test13(<32 x i32> %a, i32 %b) #0 {
entry:
  %0 = load <32 x i32>, ptr @c, align 128
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vdmpyhb.dv.acc(<32 x i32> %0, <32 x i32> %a, i32 %b)
  store <32 x i32> %1, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test14:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h += vmpy(v{{[0-9]+}}.ub,r{{[0-9]+}}.b)
define void @test14(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = load <32 x i32>, ptr @c, align 128
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vmpybus.acc(<32 x i32> %0, <16 x i32> %a, i32 %b)
  store <32 x i32> %1, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test15:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h += vmpa(v{{[0-9]+}}:{{[0-9]+}}.ub,r{{[0-9]+}}.b)
define void @test15(<32 x i32> %a, i32 %b) #0 {
entry:
  %0 = load <32 x i32>, ptr @c, align 128
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %0, <32 x i32> %a, i32 %b)
  store <32 x i32> %1, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test16:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w += vmpa(v{{[0-9]+}}:{{[0-9]+}}.h,r{{[0-9]+}}.b)
define void @test16(<32 x i32> %a, i32 %b) #0 {
entry:
  %0 = load <32 x i32>, ptr @c, align 128
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vmpahb.acc(<32 x i32> %0, <32 x i32> %a, i32 %b)
  store <32 x i32> %1, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test17:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w += vmpy(v{{[0-9]+}}.h,r{{[0-9]+}}.h):sat
define void @test17(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = load <32 x i32>, ptr @c, align 128
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vmpyhsat.acc(<32 x i32> %0, <16 x i32> %a, i32 %b)
  store <32 x i32> %1, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test18:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.uw += vmpy(v{{[0-9]+}}.uh,r{{[0-9]+}}.uh)
define void @test18(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = load <32 x i32>, ptr @c, align 128
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vmpyuh.acc(<32 x i32> %0, <16 x i32> %a, i32 %b)
  store <32 x i32> %1, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test19:
; CHECK: v{{[0-9]+}}.w += vmpyi(v{{[0-9]+}}.w,r{{[0-9]+}}.b)
define void @test19(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = load <16 x i32>, ptr @d, align 64
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %0, <16 x i32> %a, i32 %b)
  store <16 x i32> %1, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test20:
; CHECK: v{{[0-9]+}}.w += vmpyi(v{{[0-9]+}}.w,r{{[0-9]+}}.h)
define void @test20(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = load <16 x i32>, ptr @d, align 64
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwh.acc(<16 x i32> %0, <16 x i32> %a, i32 %b)
  store <16 x i32> %1, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test21:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.uw += vdsad(v{{[0-9]+}}:{{[0-9]+}}.uh,r{{[0-9]+}}.uh)
define void @test21(<32 x i32> %a, i32 %b) #0 {
entry:
  %0 = load <32 x i32>, ptr @c, align 128
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vdsaduh.acc(<32 x i32> %0, <32 x i32> %a, i32 %b)
  store <32 x i32> %1, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test22:
; CHECK: v{{[0-9]+}}.h += vmpyi(v{{[0-9]+}}.h,r{{[0-9]+}}.b)
define void @test22(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = load <16 x i32>, ptr @d, align 64
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vmpyihb.acc(<16 x i32> %0, <16 x i32> %a, i32 %b)
  store <16 x i32> %1, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test23:
; CHECK: v{{[0-9]+}}.w += vasl(v{{[0-9]+}}.w,r{{[0-9]+}})
define void @test23(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = load <16 x i32>, ptr @d, align 64
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vaslw.acc(<16 x i32> %0, <16 x i32> %a, i32 %b)
  store <16 x i32> %1, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test24:
; CHECK: v{{[0-9]+}}.w += vasr(v{{[0-9]+}}.w,r{{[0-9]+}})
define void @test24(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = load <16 x i32>, ptr @d, align 64
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vasrw.acc(<16 x i32> %0, <16 x i32> %a, i32 %b)
  store <16 x i32> %1, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test25:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.uh += vmpy(v{{[0-9]+}}.ub,r{{[0-9]+}}.ub)
define void @test25(<16 x i32> %a, i32 %b) #0 {
entry:
  %0 = load <32 x i32>, ptr @c, align 128
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vmpyub.acc(<32 x i32> %0, <16 x i32> %a, i32 %b)
  store <32 x i32> %1, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test26:
; CHECK: v{{[0-9]+}}.w += vdmpy(v{{[0-9]+}}.h,v{{[0-9]+}}.h):sat
define void @test26(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = load <16 x i32>, ptr @d, align 64
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vdmpyhvsat.acc(<16 x i32> %0, <16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %1, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test27:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h += vmpy(v{{[0-9]+}}.ub,v{{[0-9]+}}.b)
define void @test27(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = load <32 x i32>, ptr @c, align 128
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vmpybusv.acc(<32 x i32> %0, <16 x i32> %a, <16 x i32> %b)
  store <32 x i32> %1, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test28:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h += vmpy(v{{[0-9]+}}.b,v{{[0-9]+}}.b)
define void @test28(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = load <32 x i32>, ptr @c, align 128
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vmpybv.acc(<32 x i32> %0, <16 x i32> %a, <16 x i32> %b)
  store <32 x i32> %1, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test29:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w += vmpy(v{{[0-9]+}}.h,v{{[0-9]+}}.uh)
define void @test29(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = load <32 x i32>, ptr @c, align 128
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vmpyhus.acc(<32 x i32> %0, <16 x i32> %a, <16 x i32> %b)
  store <32 x i32> %1, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test30:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w += vmpy(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define void @test30(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = load <32 x i32>, ptr @c, align 128
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vmpyhv.acc(<32 x i32> %0, <16 x i32> %a, <16 x i32> %b)
  store <32 x i32> %1, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test31:
; CHECK: v{{[0-9]+}}.w += vmpyie(v{{[0-9]+}}.w,v{{[0-9]+}}.h)
define void @test31(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = load <16 x i32>, ptr @d, align 64
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiewh.acc(<16 x i32> %0, <16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %1, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test32:
; CHECK: v{{[0-9]+}}.w += vmpyie(v{{[0-9]+}}.w,v{{[0-9]+}}.uh)
define void @test32(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = load <16 x i32>, ptr @d, align 64
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiewuh.acc(<16 x i32> %0, <16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %1, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test33:
; CHECK: v{{[0-9]+}}.h += vmpyi(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
define void @test33(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = load <16 x i32>, ptr @d, align 64
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vmpyih.acc(<16 x i32> %0, <16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %1, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test34:
; CHECK: v{{[0-9]+}}.w += vmpyo(v{{[0-9]+}}.w,v{{[0-9]+}}.h):<<1:rnd:sat:shift
define void @test34(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = load <16 x i32>, ptr @d, align 64
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vmpyowh.rnd.sacc(<16 x i32> %0, <16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %1, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test35:
; CHECK: v{{[0-9]+}}.w += vmpyo(v{{[0-9]+}}.w,v{{[0-9]+}}.h):<<1:sat:shift
define void @test35(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = load <16 x i32>, ptr @d, align 64
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vmpyowh.sacc(<16 x i32> %0, <16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %1, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test36:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.uh += vmpy(v{{[0-9]+}}.ub,v{{[0-9]+}}.ub)
define void @test36(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = load <32 x i32>, ptr @c, align 128
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vmpyubv.acc(<32 x i32> %0, <16 x i32> %a, <16 x i32> %b)
  store <32 x i32> %1, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test37:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.uw += vmpy(v{{[0-9]+}}.uh,v{{[0-9]+}}.uh)
define void @test37(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = load <32 x i32>, ptr @c, align 128
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vmpyuhv.acc(<32 x i32> %0, <16 x i32> %a, <16 x i32> %b)
  store <32 x i32> %1, ptr @c, align 128
  ret void
}

; CHECK-LABEL: test38:
; CHECK: v{{[0-9]+}}.w += vrmpy(v{{[0-9]+}}.ub,v{{[0-9]+}}.b)
define void @test38(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = load <16 x i32>, ptr @d, align 64
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vrmpybusv.acc(<16 x i32> %0, <16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %1, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test39:
; CHECK: v{{[0-9]+}}.w += vrmpy(v{{[0-9]+}}.b,v{{[0-9]+}}.b)
define void @test39(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = load <16 x i32>, ptr @d, align 64
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vrmpybv.acc(<16 x i32> %0, <16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %1, ptr @d, align 64
  ret void
}

; CHECK-LABEL: test40:
; CHECK: v{{[0-9]+}}.uw += vrmpy(v{{[0-9]+}}.ub,v{{[0-9]+}}.ub)
define void @test40(<16 x i32> %a, <16 x i32> %b) #0 {
entry:
  %0 = load <16 x i32>, ptr @d, align 64
  %1 = tail call <16 x i32> @llvm.hexagon.V6.vrmpyubv.acc(<16 x i32> %0, <16 x i32> %a, <16 x i32> %b)
  store <16 x i32> %1, ptr @d, align 64
  ret void
}

declare <32 x i32> @llvm.hexagon.V6.vtmpyb.acc(<32 x i32>, <32 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vtmpybus.acc(<32 x i32>, <32 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vtmpyhb.acc(<32 x i32>, <32 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vdmpyhb.acc(<16 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vrmpyub.acc(<16 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vrmpybus.acc(<16 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vdmpybus.acc(<16 x i32>, <16 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32>, <32 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vdmpyhsusat.acc(<16 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vdmpyhsuisat.acc(<16 x i32>, <32 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vdmpyhisat.acc(<16 x i32>, <32 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vdmpyhsat.acc(<16 x i32>, <16 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vdmpyhb.dv.acc(<32 x i32>, <32 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vmpybus.acc(<32 x i32>, <16 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32>, <32 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vmpahb.acc(<32 x i32>, <32 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vmpyhsat.acc(<32 x i32>, <16 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vmpyuh.acc(<32 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vmpyiwh.acc(<16 x i32>, <16 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vdsaduh.acc(<32 x i32>, <32 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vmpyihb.acc(<16 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vaslw.acc(<16 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vasrw.acc(<16 x i32>, <16 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vmpyub.acc(<32 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vdmpyhvsat.acc(<16 x i32>, <16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vmpybusv.acc(<32 x i32>, <16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vmpybv.acc(<32 x i32>, <16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vmpyhus.acc(<32 x i32>, <16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vmpyhv.acc(<32 x i32>, <16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vmpyiewh.acc(<16 x i32>, <16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vmpyiewuh.acc(<16 x i32>, <16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vmpyih.acc(<16 x i32>, <16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vmpyowh.rnd.sacc(<16 x i32>, <16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vmpyowh.sacc(<16 x i32>, <16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vmpyubv.acc(<32 x i32>, <16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vmpyuhv.acc(<32 x i32>, <16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vrmpybusv.acc(<16 x i32>, <16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vrmpybv.acc(<16 x i32>, <16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vrmpyubv.acc(<16 x i32>, <16 x i32>, <16 x i32>) #0

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvx60,+hvx-length64b" }
