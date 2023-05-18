; RUN: llc -march=hexagon -O0 < %s | FileCheck %s
; RUN: llc -march=hexagon -O0 < %s | FileCheck -check-prefix=CHECK-CALL %s
; Hexagon Programmer's Reference Manual 11.10.6 XTYPE/PERM

; CHECK-CALL-NOT: call

; Saturate
declare i32 @llvm.hexagon.A2.sat(i64)
define i32 @A2_sat(i64 %a) {
  %z = call i32 @llvm.hexagon.A2.sat(i64 %a)
  ret i32 %z
}
; CHECK: = sat({{.*}})

declare i32 @llvm.hexagon.A2.sath(i32)
define i32 @A2_sath(i32 %a) {
  %z = call i32 @llvm.hexagon.A2.sath(i32 %a)
  ret i32 %z
}
; CHECK: = sath({{.*}})

declare i32 @llvm.hexagon.A2.satuh(i32)
define i32 @A2_satuh(i32 %a) {
  %z = call i32 @llvm.hexagon.A2.satuh(i32 %a)
  ret i32 %z
}
; CHECK: = satuh({{.*}})

declare i32 @llvm.hexagon.A2.satub(i32)
define i32 @A2_satub(i32 %a) {
  %z = call i32 @llvm.hexagon.A2.satub(i32 %a)
  ret i32 %z
}
; CHECK: = satub({{.*}})

declare i32 @llvm.hexagon.A2.satb(i32)
define i32 @A2_satb(i32 %a) {
  %z = call i32 @llvm.hexagon.A2.satb(i32 %a)
  ret i32 %z
}
; CHECK: = satb({{.*}})

; Swizzle bytes
declare i32 @llvm.hexagon.A2.swiz(i32)
define i32 @A2_swiz(i32 %a) {
  %z = call i32 @llvm.hexagon.A2.swiz(i32 %a)
  ret i32 %z
}
; CHECK: = swiz({{.*}})

; Vector round and pack
declare i32 @llvm.hexagon.S2.vrndpackwh(i64)
define i32 @S2_vrndpackwh(i64 %a) {
  %z = call i32 @llvm.hexagon.S2.vrndpackwh(i64 %a)
  ret i32 %z
}
; CHECK: = vrndwh({{.*}})

declare i32 @llvm.hexagon.S2.vrndpackwhs(i64)
define i32 @S2_vrndpackwhs(i64 %a) {
  %z = call i32 @llvm.hexagon.S2.vrndpackwhs(i64 %a)
  ret i32 %z
}
; CHECK: = vrndwh({{.*}}):sat

; Vector saturate and pack
declare i32 @llvm.hexagon.S2.vsathub(i64)
define i32 @S2_vsathub(i64 %a) {
  %z = call i32 @llvm.hexagon.S2.vsathub(i64 %a)
  ret i32 %z
}
; CHECK: = vsathub({{.*}})

declare i32 @llvm.hexagon.S2.vsatwh(i64)
define i32 @S2_vsatwh(i64 %a) {
  %z = call i32 @llvm.hexagon.S2.vsatwh(i64 %a)
  ret i32 %z
}
; CHECK: = vsatwh({{.*}})

declare i32 @llvm.hexagon.S2.vsatwuh(i64)
define i32 @S2_vsatwuh(i64 %a) {
  %z = call i32 @llvm.hexagon.S2.vsatwuh(i64 %a)
  ret i32 %z
}
; CHECK: = vsatwuh({{.*}})

declare i32 @llvm.hexagon.S2.vsathb(i64)
define i32 @S2_vsathb(i64 %a) {
  %z = call i32 @llvm.hexagon.S2.vsathb(i64 %a)
  ret i32 %z
}
; CHECK: = vsathb({{.*}})

declare i32 @llvm.hexagon.S2.svsathb(i32)
define i32 @S2_svsathb(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.svsathb(i32 %a)
  ret i32 %z
}
; CHECK: = vsathb({{.*}})

declare i32 @llvm.hexagon.S2.svsathub(i32)
define i32 @S2_svsathub(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.svsathub(i32 %a)
  ret i32 %z
}
; CHECK: = vsathub({{.*}})

; Vector saturate without pack
declare i64 @llvm.hexagon.S2.vsathub.nopack(i64)
define i64 @S2_vsathub_nopack(i64 %a) {
  %z = call i64 @llvm.hexagon.S2.vsathub.nopack(i64 %a)
  ret i64 %z
}
; CHECK: = vsathub({{.*}})

declare i64 @llvm.hexagon.S2.vsatwuh.nopack(i64)
define i64 @S2_vsatwuh_nopack(i64 %a) {
  %z = call i64 @llvm.hexagon.S2.vsatwuh.nopack(i64 %a)
  ret i64 %z
}
; CHECK: = vsatwuh({{.*}})

declare i64 @llvm.hexagon.S2.vsatwh.nopack(i64)
define i64 @S2_vsatwh_nopack(i64 %a) {
  %z = call i64 @llvm.hexagon.S2.vsatwh.nopack(i64 %a)
  ret i64 %z
}
; CHECK: = vsatwh({{.*}})

declare i64 @llvm.hexagon.S2.vsathb.nopack(i64)
define i64 @S2_vsathb_nopack(i64 %a) {
  %z = call i64 @llvm.hexagon.S2.vsathb.nopack(i64 %a)
  ret i64 %z
}
; CHECK: = vsathb({{.*}})

; Vector shuffle
declare i64 @llvm.hexagon.S2.shuffeb(i64, i64)
define i64 @S2_shuffeb(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.shuffeb(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: = shuffeb({{.*}},{{.*}})

declare i64 @llvm.hexagon.S2.shuffob(i64, i64)
define i64 @S2_shuffob(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.shuffob(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: = shuffob({{.*}},{{.*}})

declare i64 @llvm.hexagon.S2.shuffeh(i64, i64)
define i64 @S2_shuffeh(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.shuffeh(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: = shuffeh({{.*}},{{.*}})

declare i64 @llvm.hexagon.S2.shuffoh(i64, i64)
define i64 @S2_shuffoh(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.shuffoh(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: = shuffoh({{.*}},{{.*}})

; Vector splat bytes
declare i32 @llvm.hexagon.S2.vsplatrb(i32)
define i32 @S2_vsplatrb(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.vsplatrb(i32 %a)
  ret i32 %z
}
; CHECK: = vsplatb({{.*}})

; Vector splat halfwords
declare i64 @llvm.hexagon.S2.vsplatrh(i32)
define i64 @S2_vsplatrh(i32 %a) {
  %z = call i64 @llvm.hexagon.S2.vsplatrh(i32 %a)
  ret i64 %z
}
; CHECK: = vsplath({{.*}})

; Vector splice
declare i64 @llvm.hexagon.S2.vspliceib(i64, i64, i32)
define i64 @S2_vspliceib(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.vspliceib(i64 %a, i64 %b, i32 0)
  ret i64 %z
}
; CHECK: = vspliceb({{.*}},{{.*}},#0)

declare i64 @llvm.hexagon.S2.vsplicerb(i64, i64, i32)
define i64 @S2_vsplicerb(i64 %a, i64 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.S2.vsplicerb(i64 %a, i64 %b, i32 %c)
  ret i64 %z
}
; CHECK: = vspliceb({{.*}},{{.*}},{{.*}})

; Vector sign extend
declare i64 @llvm.hexagon.S2.vsxtbh(i32)
define i64 @S2_vsxtbh(i32 %a) {
  %z = call i64 @llvm.hexagon.S2.vsxtbh(i32 %a)
  ret i64 %z
}
; CHECK: = vsxtbh({{.*}})

declare i64 @llvm.hexagon.S2.vsxthw(i32)
define i64 @S2_vsxthw(i32 %a) {
  %z = call i64 @llvm.hexagon.S2.vsxthw(i32 %a)
  ret i64 %z
}
; CHECK: = vsxthw({{.*}})

; Vector truncate
declare i32 @llvm.hexagon.S2.vtrunohb(i64)
define i32 @S2_vtrunohb(i64 %a) {
  %z = call i32 @llvm.hexagon.S2.vtrunohb(i64 %a)
  ret i32 %z
}
; CHECK: = vtrunohb({{.*}})

declare i32 @llvm.hexagon.S2.vtrunehb(i64)
define i32 @S2_vtrunehb(i64 %a) {
  %z = call i32 @llvm.hexagon.S2.vtrunehb(i64 %a)
  ret i32 %z
}
; CHECK: = vtrunehb({{.*}})

declare i64 @llvm.hexagon.S2.vtrunowh(i64, i64)
define i64 @S2_vtrunowh(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.vtrunowh(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: = vtrunowh({{.*}},{{.*}})

declare i64 @llvm.hexagon.S2.vtrunewh(i64, i64)
define i64 @S2_vtrunewh(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.vtrunewh(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: = vtrunewh({{.*}},{{.*}})

; Vector zero extend
declare i64 @llvm.hexagon.S2.vzxtbh(i32)
define i64 @S2_vzxtbh(i32 %a) {
  %z = call i64 @llvm.hexagon.S2.vzxtbh(i32 %a)
  ret i64 %z
}
; CHECK: = vzxtbh({{.*}})

declare i64 @llvm.hexagon.S2.vzxthw(i32)
define i64 @S2_vzxthw(i32 %a) {
  %z = call i64 @llvm.hexagon.S2.vzxthw(i32 %a)
  ret i64 %z
}
; CHECK: = vzxthw({{.*}})
