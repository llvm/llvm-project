; RUN: llc -mtriple=hexagon < %s | FileCheck %s

@c = external global <64 x i32>
@d = external global <32 x i32>

; CHECK-LABEL: test1:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h += vtmpy(v{{[0-9]+}}:{{[0-9]+}}.b,r{{[0-9]+}}.b)
define void @test1(<64 x i32> %a, i32 %b) #0 {
entry:
  %a.addr = alloca <64 x i32>, align 256
  %b.addr = alloca i32, align 4
  store <64 x i32> %a, ptr %a.addr, align 256
  store i32 %b, ptr %b.addr, align 4
  %0 = load <64 x i32>, ptr @c, align 256
  %1 = load <64 x i32>, ptr %a.addr, align 256
  %2 = load i32, ptr %b.addr, align 4
  %3 = call <64 x i32> @llvm.hexagon.V6.vtmpyb.acc.128B(<64 x i32> %0, <64 x i32> %1, i32 %2)
  store <64 x i32> %3, ptr @c, align 256
  ret void
}

; CHECK-LABEL: test2:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h += vtmpy(v{{[0-9]+}}:{{[0-9]+}}.ub,r{{[0-9]+}}.b)
define void @test2(<64 x i32> %a, i32 %b) #0 {
entry:
  %a.addr = alloca <64 x i32>, align 256
  %b.addr = alloca i32, align 4
  store <64 x i32> %a, ptr %a.addr, align 256
  store i32 %b, ptr %b.addr, align 4
  %0 = load <64 x i32>, ptr @c, align 256
  %1 = load <64 x i32>, ptr %a.addr, align 256
  %2 = load i32, ptr %b.addr, align 4
  %3 = call <64 x i32> @llvm.hexagon.V6.vtmpybus.acc.128B(<64 x i32> %0, <64 x i32> %1, i32 %2)
  store <64 x i32> %3, ptr @c, align 256
  ret void
}

; CHECK-LABEL: test3:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w += vtmpy(v{{[0-9]+}}:{{[0-9]+}}.h,r{{[0-9]+}}.b)
define void @test3(<64 x i32> %a, i32 %b) #0 {
entry:
  %a.addr = alloca <64 x i32>, align 256
  %b.addr = alloca i32, align 4
  store <64 x i32> %a, ptr %a.addr, align 256
  store i32 %b, ptr %b.addr, align 4
  %0 = load <64 x i32>, ptr @c, align 256
  %1 = load <64 x i32>, ptr %a.addr, align 256
  %2 = load i32, ptr %b.addr, align 4
  %3 = call <64 x i32> @llvm.hexagon.V6.vtmpyhb.acc.128B(<64 x i32> %0, <64 x i32> %1, i32 %2)
  store <64 x i32> %3, ptr @c, align 256
  ret void
}

; CHECK-LABEL: test4:
; CHECK: v{{[0-9]+}}.w += vdmpy(v{{[0-9]+}}.h,r{{[0-9]+}}.b)
define void @test4(<32 x i32> %a, i32 %b) #0 {
entry:
  %a.addr = alloca <32 x i32>, align 128
  %b.addr = alloca i32, align 4
  store <32 x i32> %a, ptr %a.addr, align 128
  store i32 %b, ptr %b.addr, align 4
  %0 = load <32 x i32>, ptr @d, align 128
  %1 = load <32 x i32>, ptr %a.addr, align 128
  %2 = load i32, ptr %b.addr, align 4
  %3 = call <32 x i32> @llvm.hexagon.V6.vdmpyhb.acc.128B(<32 x i32> %0, <32 x i32> %1, i32 %2)
  store <32 x i32> %3, ptr @d, align 128
  ret void
}

; CHECK-LABEL: test5:
; CHECK: v{{[0-9]+}}.uw += vrmpy(v{{[0-9]+}}.ub,r{{[0-9]+}}.ub)
define void @test5(<32 x i32> %a, i32 %b) #0 {
entry:
  %a.addr = alloca <32 x i32>, align 128
  %b.addr = alloca i32, align 4
  store <32 x i32> %a, ptr %a.addr, align 128
  store i32 %b, ptr %b.addr, align 4
  %0 = load <32 x i32>, ptr @d, align 128
  %1 = load <32 x i32>, ptr %a.addr, align 128
  %2 = load i32, ptr %b.addr, align 4
  %3 = call <32 x i32> @llvm.hexagon.V6.vrmpyub.acc.128B(<32 x i32> %0, <32 x i32> %1, i32 %2)
  store <32 x i32> %3, ptr @d, align 128
  ret void
}

; CHECK-LABEL: test6:
; CHECK: v{{[0-9]+}}.w += vrmpy(v{{[0-9]+}}.ub,r{{[0-9]+}}.b)
define void @test6(<32 x i32> %a, i32 %b) #0 {
entry:
  %a.addr = alloca <32 x i32>, align 128
  %b.addr = alloca i32, align 4
  store <32 x i32> %a, ptr %a.addr, align 128
  store i32 %b, ptr %b.addr, align 4
  %0 = load <32 x i32>, ptr @d, align 128
  %1 = load <32 x i32>, ptr %a.addr, align 128
  %2 = load i32, ptr %b.addr, align 4
  %3 = call <32 x i32> @llvm.hexagon.V6.vrmpybus.acc.128B(<32 x i32> %0, <32 x i32> %1, i32 %2)
  store <32 x i32> %3, ptr @d, align 128
  ret void
}

; CHECK-LABEL: test7:
; CHECK: v{{[0-9]+}}.h += vdmpy(v{{[0-9]+}}.ub,r{{[0-9]+}}.b)
define void @test7(<32 x i32> %a, i32 %b) #0 {
entry:
  %a.addr = alloca <32 x i32>, align 128
  %b.addr = alloca i32, align 4
  store <32 x i32> %a, ptr %a.addr, align 128
  store i32 %b, ptr %b.addr, align 4
  %0 = load <32 x i32>, ptr @d, align 128
  %1 = load <32 x i32>, ptr %a.addr, align 128
  %2 = load i32, ptr %b.addr, align 4
  %3 = call <32 x i32> @llvm.hexagon.V6.vdmpybus.acc.128B(<32 x i32> %0, <32 x i32> %1, i32 %2)
  store <32 x i32> %3, ptr @d, align 128
  ret void
}

; CHECK-LABEL: test8:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h += vdmpy(v{{[0-9]+}}:{{[0-9]+}}.ub,r{{[0-9]+}}.b)
define void @test8(<64 x i32> %a, i32 %b) #0 {
entry:
  %a.addr = alloca <64 x i32>, align 256
  %b.addr = alloca i32, align 4
  store <64 x i32> %a, ptr %a.addr, align 256
  store i32 %b, ptr %b.addr, align 4
  %0 = load <64 x i32>, ptr @c, align 256
  %1 = load <64 x i32>, ptr %a.addr, align 256
  %2 = load i32, ptr %b.addr, align 4
  %3 = call <64 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc.128B(<64 x i32> %0, <64 x i32> %1, i32 %2)
  store <64 x i32> %3, ptr @c, align 256
  ret void
}

; CHECK-LABEL: test9:
; CHECK: v{{[0-9]+}}.w += vdmpy(v{{[0-9]+}}.h,r{{[0-9]+}}.uh):sat
define void @test9(<32 x i32> %a, i32 %b) #0 {
entry:
  %a.addr = alloca <32 x i32>, align 128
  %b.addr = alloca i32, align 4
  store <32 x i32> %a, ptr %a.addr, align 128
  store i32 %b, ptr %b.addr, align 4
  %0 = load <32 x i32>, ptr @d, align 128
  %1 = load <32 x i32>, ptr %a.addr, align 128
  %2 = load i32, ptr %b.addr, align 4
  %3 = call <32 x i32> @llvm.hexagon.V6.vdmpyhsusat.acc.128B(<32 x i32> %0, <32 x i32> %1, i32 %2)
  store <32 x i32> %3, ptr @d, align 128
  ret void
}

; CHECK-LABEL: test10:
; CHECK: v{{[0-9]+}}.w += vdmpy(v{{[0-9]+}}:{{[0-9]+}}.h,r{{[0-9]+}}.uh,#1):sat
define void @test10(<64 x i32> %a, i32 %b) #0 {
entry:
  %a.addr = alloca <64 x i32>, align 256
  %b.addr = alloca i32, align 4
  store <64 x i32> %a, ptr %a.addr, align 256
  store i32 %b, ptr %b.addr, align 4
  %0 = load <32 x i32>, ptr @d, align 128
  %1 = load <64 x i32>, ptr %a.addr, align 256
  %2 = load i32, ptr %b.addr, align 4
  %3 = call <32 x i32> @llvm.hexagon.V6.vdmpyhsuisat.acc.128B(<32 x i32> %0, <64 x i32> %1, i32 %2)
  store <32 x i32> %3, ptr @d, align 128
  ret void
}

; CHECK-LABEL: test11:
; CHECK: v{{[0-9]+}}.w += vdmpy(v{{[0-9]+}}:{{[0-9]+}}.h,r{{[0-9]+}}.h):sat
define void @test11(<64 x i32> %a, i32 %b) #0 {
entry:
  %a.addr = alloca <64 x i32>, align 256
  %b.addr = alloca i32, align 4
  store <64 x i32> %a, ptr %a.addr, align 256
  store i32 %b, ptr %b.addr, align 4
  %0 = load <32 x i32>, ptr @d, align 128
  %1 = load <64 x i32>, ptr %a.addr, align 256
  %2 = load i32, ptr %b.addr, align 4
  %3 = call <32 x i32> @llvm.hexagon.V6.vdmpyhisat.acc.128B(<32 x i32> %0, <64 x i32> %1, i32 %2)
  store <32 x i32> %3, ptr @d, align 128
  ret void
}

; CHECK-LABEL: test12:
; CHECK: v{{[0-9]+}}.w += vdmpy(v{{[0-9]+}}.h,r{{[0-9]+}}.h):sat
define void @test12(<32 x i32> %a, i32 %b) #0 {
entry:
  %a.addr = alloca <32 x i32>, align 128
  %b.addr = alloca i32, align 4
  store <32 x i32> %a, ptr %a.addr, align 128
  store i32 %b, ptr %b.addr, align 4
  %0 = load <32 x i32>, ptr @d, align 128
  %1 = load <32 x i32>, ptr %a.addr, align 128
  %2 = load i32, ptr %b.addr, align 4
  %3 = call <32 x i32> @llvm.hexagon.V6.vdmpyhsat.acc.128B(<32 x i32> %0, <32 x i32> %1, i32 %2)
  store <32 x i32> %3, ptr @d, align 128
  ret void
}

; CHECK-LABEL: test13:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w += vdmpy(v{{[0-9]+}}:{{[0-9]+}}.h,r{{[0-9]+}}.b)
define void @test13(<64 x i32> %a, i32 %b) #0 {
entry:
  %a.addr = alloca <64 x i32>, align 256
  %b.addr = alloca i32, align 4
  store <64 x i32> %a, ptr %a.addr, align 256
  store i32 %b, ptr %b.addr, align 4
  %0 = load <64 x i32>, ptr @c, align 256
  %1 = load <64 x i32>, ptr %a.addr, align 256
  %2 = load i32, ptr %b.addr, align 4
  %3 = call <64 x i32> @llvm.hexagon.V6.vdmpyhb.dv.acc.128B(<64 x i32> %0, <64 x i32> %1, i32 %2)
  store <64 x i32> %3, ptr @c, align 256
  ret void
}

; CHECK-LABEL: test14:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h += vmpy(v{{[0-9]+}}.ub,r{{[0-9]+}}.b)
define void @test14(<32 x i32> %a, i32 %b) #0 {
entry:
  %a.addr = alloca <32 x i32>, align 128
  %b.addr = alloca i32, align 4
  store <32 x i32> %a, ptr %a.addr, align 128
  store i32 %b, ptr %b.addr, align 4
  %0 = load <64 x i32>, ptr @c, align 256
  %1 = load <32 x i32>, ptr %a.addr, align 128
  %2 = load i32, ptr %b.addr, align 4
  %3 = call <64 x i32> @llvm.hexagon.V6.vmpybus.acc.128B(<64 x i32> %0, <32 x i32> %1, i32 %2)
  store <64 x i32> %3, ptr @c, align 256
  ret void
}

; CHECK-LABEL: test15:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.h += vmpa(v{{[0-9]+}}:{{[0-9]+}}.ub,r{{[0-9]+}}.b)
define void @test15(<64 x i32> %a, i32 %b) #0 {
entry:
  %a.addr = alloca <64 x i32>, align 256
  %b.addr = alloca i32, align 4
  store <64 x i32> %a, ptr %a.addr, align 256
  store i32 %b, ptr %b.addr, align 4
  %0 = load <64 x i32>, ptr @c, align 256
  %1 = load <64 x i32>, ptr %a.addr, align 256
  %2 = load i32, ptr %b.addr, align 4
  %3 = call <64 x i32> @llvm.hexagon.V6.vmpabus.acc.128B(<64 x i32> %0, <64 x i32> %1, i32 %2)
  store <64 x i32> %3, ptr @c, align 256
  ret void
}

; CHECK-LABEL: test16:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w += vmpa(v{{[0-9]+}}:{{[0-9]+}}.h,r{{[0-9]+}}.b)
define void @test16(<64 x i32> %a, i32 %b) #0 {
entry:
  %a.addr = alloca <64 x i32>, align 256
  %b.addr = alloca i32, align 4
  store <64 x i32> %a, ptr %a.addr, align 256
  store i32 %b, ptr %b.addr, align 4
  %0 = load <64 x i32>, ptr @c, align 256
  %1 = load <64 x i32>, ptr %a.addr, align 256
  %2 = load i32, ptr %b.addr, align 4
  %3 = call <64 x i32> @llvm.hexagon.V6.vmpahb.acc.128B(<64 x i32> %0, <64 x i32> %1, i32 %2)
  store <64 x i32> %3, ptr @c, align 256
  ret void
}

; CHECK-LABEL: test17:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.w += vmpy(v{{[0-9]+}}.h,r{{[0-9]+}}.h):sat
define void @test17(<32 x i32> %a, i32 %b) #0 {
entry:
  %a.addr = alloca <32 x i32>, align 128
  %b.addr = alloca i32, align 4
  store <32 x i32> %a, ptr %a.addr, align 128
  store i32 %b, ptr %b.addr, align 4
  %0 = load <64 x i32>, ptr @c, align 256
  %1 = load <32 x i32>, ptr %a.addr, align 128
  %2 = load i32, ptr %b.addr, align 4
  %3 = call <64 x i32> @llvm.hexagon.V6.vmpyhsat.acc.128B(<64 x i32> %0, <32 x i32> %1, i32 %2)
  store <64 x i32> %3, ptr @c, align 256
  ret void
}

; CHECK-LABEL: test18:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.uw += vmpy(v{{[0-9]+}}.uh,r{{[0-9]+}}.uh)
define void @test18(<32 x i32> %a, i32 %b) #0 {
entry:
  %a.addr = alloca <32 x i32>, align 128
  %b.addr = alloca i32, align 4
  store <32 x i32> %a, ptr %a.addr, align 128
  store i32 %b, ptr %b.addr, align 4
  %0 = load <64 x i32>, ptr @c, align 256
  %1 = load <32 x i32>, ptr %a.addr, align 128
  %2 = load i32, ptr %b.addr, align 4
  %3 = call <64 x i32> @llvm.hexagon.V6.vmpyuh.acc.128B(<64 x i32> %0, <32 x i32> %1, i32 %2)
  store <64 x i32> %3, ptr @c, align 256
  ret void
}

; CHECK-LABEL: test19:
; CHECK: v{{[0-9]+}}.w += vmpyi(v{{[0-9]+}}.w,r{{[0-9]+}}.b)
define void @test19(<32 x i32> %a, i32 %b) #0 {
entry:
  %a.addr = alloca <32 x i32>, align 128
  %b.addr = alloca i32, align 4
  store <32 x i32> %a, ptr %a.addr, align 128
  store i32 %b, ptr %b.addr, align 4
  %0 = load <32 x i32>, ptr @d, align 128
  %1 = load <32 x i32>, ptr %a.addr, align 128
  %2 = load i32, ptr %b.addr, align 4
  %3 = call <32 x i32> @llvm.hexagon.V6.vmpyiwb.acc.128B(<32 x i32> %0, <32 x i32> %1, i32 %2)
  store <32 x i32> %3, ptr @d, align 128
  ret void
}

; CHECK-LABEL: test20:
; CHECK: v{{[0-9]+}}.w += vmpyi(v{{[0-9]+}}.w,r{{[0-9]+}}.h)
define void @test20(<32 x i32> %a, i32 %b) #0 {
entry:
  %a.addr = alloca <32 x i32>, align 128
  %b.addr = alloca i32, align 4
  store <32 x i32> %a, ptr %a.addr, align 128
  store i32 %b, ptr %b.addr, align 4
  %0 = load <32 x i32>, ptr @d, align 128
  %1 = load <32 x i32>, ptr %a.addr, align 128
  %2 = load i32, ptr %b.addr, align 4
  %3 = call <32 x i32> @llvm.hexagon.V6.vmpyiwh.acc.128B(<32 x i32> %0, <32 x i32> %1, i32 %2)
  store <32 x i32> %3, ptr @d, align 128
  ret void
}

; CHECK-LABEL: test21:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.uw += vdsad(v{{[0-9]+}}:{{[0-9]+}}.uh,r{{[0-9]+}}.uh)
define void @test21(<64 x i32> %a, i32 %b) #0 {
entry:
  %a.addr = alloca <64 x i32>, align 256
  %b.addr = alloca i32, align 4
  store <64 x i32> %a, ptr %a.addr, align 256
  store i32 %b, ptr %b.addr, align 4
  %0 = load <64 x i32>, ptr @c, align 256
  %1 = load <64 x i32>, ptr %a.addr, align 256
  %2 = load i32, ptr %b.addr, align 4
  %3 = call <64 x i32> @llvm.hexagon.V6.vdsaduh.acc.128B(<64 x i32> %0, <64 x i32> %1, i32 %2)
  store <64 x i32> %3, ptr @c, align 256
  ret void
}

; CHECK-LABEL: test22:
; CHECK: v{{[0-9]+}}.h += vmpyi(v{{[0-9]+}}.h,r{{[0-9]+}}.b)
define void @test22(<32 x i32> %a, i32 %b) #0 {
entry:
  %a.addr = alloca <32 x i32>, align 128
  %b.addr = alloca i32, align 4
  store <32 x i32> %a, ptr %a.addr, align 128
  store i32 %b, ptr %b.addr, align 4
  %0 = load <32 x i32>, ptr @d, align 128
  %1 = load <32 x i32>, ptr %a.addr, align 128
  %2 = load i32, ptr %b.addr, align 4
  %3 = call <32 x i32> @llvm.hexagon.V6.vmpyihb.acc.128B(<32 x i32> %0, <32 x i32> %1, i32 %2)
  store <32 x i32> %3, ptr @d, align 128
  ret void
}

; CHECK-LABEL: test23:
; CHECK: v{{[0-9]+}}.w += vasl(v{{[0-9]+}}.w,r{{[0-9]+}})
define void @test23(<32 x i32> %a, i32 %b) #0 {
entry:
  %a.addr = alloca <32 x i32>, align 128
  %b.addr = alloca i32, align 4
  store <32 x i32> %a, ptr %a.addr, align 128
  store i32 %b, ptr %b.addr, align 4
  %0 = load <32 x i32>, ptr @d, align 128
  %1 = load <32 x i32>, ptr %a.addr, align 128
  %2 = load i32, ptr %b.addr, align 4
  %3 = call <32 x i32> @llvm.hexagon.V6.vaslw.acc.128B(<32 x i32> %0, <32 x i32> %1, i32 %2)
  store <32 x i32> %3, ptr @d, align 128
  ret void
}

; CHECK-LABEL: test24:
; CHECK: v{{[0-9]+}}.w += vasr(v{{[0-9]+}}.w,r{{[0-9]+}})
define void @test24(<32 x i32> %a, i32 %b) #0 {
entry:
  %a.addr = alloca <32 x i32>, align 128
  %b.addr = alloca i32, align 4
  store <32 x i32> %a, ptr %a.addr, align 128
  store i32 %b, ptr %b.addr, align 4
  %0 = load <32 x i32>, ptr @d, align 128
  %1 = load <32 x i32>, ptr %a.addr, align 128
  %2 = load i32, ptr %b.addr, align 4
  %3 = call <32 x i32> @llvm.hexagon.V6.vasrw.acc.128B(<32 x i32> %0, <32 x i32> %1, i32 %2)
  store <32 x i32> %3, ptr @d, align 128
  ret void
}

; CHECK-LABEL: test25:
; CHECK: v{{[0-9]+}}:{{[0-9]+}}.uh += vmpy(v{{[0-9]+}}.ub,r{{[0-9]+}}.ub)
define void @test25(<32 x i32> %a, i32 %b) #0 {
entry:
  %a.addr = alloca <32 x i32>, align 128
  %b.addr = alloca i32, align 4
  store <32 x i32> %a, ptr %a.addr, align 128
  store i32 %b, ptr %b.addr, align 4
  %0 = load <64 x i32>, ptr @c, align 256
  %1 = load <32 x i32>, ptr %a.addr, align 128
  %2 = load i32, ptr %b.addr, align 4
  %3 = call <64 x i32> @llvm.hexagon.V6.vmpyub.acc.128B(<64 x i32> %0, <32 x i32> %1, i32 %2)
  store <64 x i32> %3, ptr @c, align 256
  ret void
}

declare <64 x i32> @llvm.hexagon.V6.vtmpyb.acc.128B(<64 x i32>, <64 x i32>, i32) #0
declare <64 x i32> @llvm.hexagon.V6.vtmpybus.acc.128B(<64 x i32>, <64 x i32>, i32) #0
declare <64 x i32> @llvm.hexagon.V6.vtmpyhb.acc.128B(<64 x i32>, <64 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vdmpyhb.acc.128B(<32 x i32>, <32 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vrmpyub.acc.128B(<32 x i32>, <32 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vrmpybus.acc.128B(<32 x i32>, <32 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vdmpybus.acc.128B(<32 x i32>, <32 x i32>, i32) #0
declare <64 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc.128B(<64 x i32>, <64 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vdmpyhsusat.acc.128B(<32 x i32>, <32 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vdmpyhsuisat.acc.128B(<32 x i32>, <64 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vdmpyhisat.acc.128B(<32 x i32>, <64 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vdmpyhsat.acc.128B(<32 x i32>, <32 x i32>, i32) #0
declare <64 x i32> @llvm.hexagon.V6.vdmpyhb.dv.acc.128B(<64 x i32>, <64 x i32>, i32) #0
declare <64 x i32> @llvm.hexagon.V6.vmpybus.acc.128B(<64 x i32>, <32 x i32>, i32) #0
declare <64 x i32> @llvm.hexagon.V6.vmpabus.acc.128B(<64 x i32>, <64 x i32>, i32) #0
declare <64 x i32> @llvm.hexagon.V6.vmpahb.acc.128B(<64 x i32>, <64 x i32>, i32) #0
declare <64 x i32> @llvm.hexagon.V6.vmpyhsat.acc.128B(<64 x i32>, <32 x i32>, i32) #0
declare <64 x i32> @llvm.hexagon.V6.vmpyuh.acc.128B(<64 x i32>, <32 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vmpyiwb.acc.128B(<32 x i32>, <32 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vmpyiwh.acc.128B(<32 x i32>, <32 x i32>, i32) #0
declare <64 x i32> @llvm.hexagon.V6.vdsaduh.acc.128B(<64 x i32>, <64 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vmpyihb.acc.128B(<32 x i32>, <32 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vaslw.acc.128B(<32 x i32>, <32 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vasrw.acc.128B(<32 x i32>, <32 x i32>, i32) #0
declare <64 x i32> @llvm.hexagon.V6.vmpyub.acc.128B(<64 x i32>, <32 x i32>, i32) #0

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b" }
