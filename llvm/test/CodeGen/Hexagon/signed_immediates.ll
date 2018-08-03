; RUN: llc -march=hexagon < %s | FileCheck %s

; s4_0Imm
; CHECK: memb(r0++#-1) = r1
define i8* @foo1(i8* %a, i8 %b)  {
  store i8 %b, i8* %a
  %c = getelementptr i8, i8* %a, i32 -1
  ret i8* %c
}

; s4_1Imm
; CHECK: memh(r0++#-2) = r1
define i16* @foo2(i16* %a, i16 %b)  {
  store i16 %b, i16* %a
  %c = getelementptr i16, i16* %a, i32 -1
  ret i16* %c
}

; s4_2Imm
; CHECK: memw(r0++#-4) = r1
define i32* @foo3(i32* %a, i32 %b)  {
  store i32 %b, i32* %a
  %c = getelementptr i32, i32* %a, i32 -1
  ret i32* %c
}

; s4_3Imm
; CHECK: memd(r0++#-8) = r3:2
define i64* @foo4(i64* %a, i64 %b)  {
  store i64 %b, i64* %a
  %c = getelementptr i64, i64* %a, i32 -1
  ret i64* %c
}

; s6Ext
; CHECK: if (p0.new) memw(r0+#0) = #-1
define void @foo5(i32* %a, i1 %b) {
br i1 %b, label %x, label %y
x:
  store i32 -1, i32* %a
  ret void
y:
  ret void
}

; s10Ext
; CHECK: p0 = cmp.eq(r0,#-1)
define i1 @foo7(i32 %a) {
  %b = icmp eq i32 %a, -1
  ret i1 %b
}

; s11_0Ext
; CHECK: memb(r0+#-1) = r1
define void @foo8(i8* %a, i8 %b) {
  %c = getelementptr i8, i8* %a, i32 -1
  store i8 %b, i8* %c
  ret void
}

; s11_1Ext
; CHECK: memh(r0+#-2) = r1
define void @foo9(i16* %a, i16 %b) {
  %c = getelementptr i16, i16* %a, i32 -1
  store i16 %b, i16* %c
  ret void
}

; s11_2Ext
; CHECK: memw(r0+#-4) = r1
define void @foo10(i32* %a, i32 %b) {
  %c = getelementptr i32, i32* %a, i32 -1
  store i32 %b, i32* %c
  ret void
}

; s11_3Ext
; CHECK: memd(r0+#-8) = r3:2
define void @foo11(i64* %a, i64 %b) {
  %c = getelementptr i64, i64* %a, i32 -1
  store i64 %b, i64* %c
  ret void
}

; s12Ext
; CHECK: r1 = mux(p0,#-1,r0)
define i32 @foo12(i32 %a, i1 %b) {
  br i1 %b, label %x, label %y
x:
  ret i32 -1
y:
  ret i32 %a
}

; s16Ext
; CHECK: r0 = #-2
define i32 @foo13() {
  ret i32 -2
}
