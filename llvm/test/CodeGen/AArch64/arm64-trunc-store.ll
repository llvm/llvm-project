; RUN: llc < %s -mtriple=arm64-apple-ios7.0 -disable-post-ra | FileCheck %s

define void @bar(<8 x i16> %arg, <8 x i8>* %p) nounwind {
; CHECK-LABEL: bar:
; CHECK: xtn.8b v[[REG:[0-9]+]], v0
; CHECK-NEXT: str d[[REG]], [x0]
; CHECK-NEXT: ret
  %tmp = trunc <8 x i16> %arg to <8 x i8>
  store <8 x i8> %tmp, <8 x i8>* %p, align 8
  ret void
}

@zptr8 = common global i8* null, align 8
@zptr16 = common global i16* null, align 8
@zptr32 = common global i32* null, align 8

define void @fct32(i32 %arg, i64 %var) {
; CHECK: fct32
; CHECK: adrp [[GLOBALPAGE:x[0-9]+]], _zptr32@GOTPAGE
; CHECK: ldr [[GLOBALOFF:x[0-9]+]], {{\[}}[[GLOBALPAGE]], _zptr32@GOTPAGEOFF]
; CHECK: ldr [[GLOBALADDR:x[0-9]+]], {{\[}}[[GLOBALOFF]]]
; w0 is %arg
; CHECK-NEXT: sub w[[OFFSETREGNUM:[0-9]+]], w0, #1
; w1 is %var truncated
; CHECK-NEXT: str w1, {{\[}}[[GLOBALADDR]], w[[OFFSETREGNUM]], sxtw #2]
; CHECK-NEXT: ret
bb:
  %.pre37 = load i32*, i32** @zptr32, align 8
  %dec = add nsw i32 %arg, -1
  %idxprom8 = sext i32 %dec to i64
  %arrayidx9 = getelementptr inbounds i32, i32* %.pre37, i64 %idxprom8
  %tmp = trunc i64 %var to i32
  store i32 %tmp, i32* %arrayidx9, align 4
  ret void
}

define void @fct16(i32 %arg, i64 %var) {
; CHECK: fct16
; CHECK: adrp [[GLOBALPAGE:x[0-9]+]], _zptr16@GOTPAGE
; CHECK: ldr [[GLOBALOFF:x[0-9]+]], {{\[}}[[GLOBALPAGE]], _zptr16@GOTPAGEOFF]
; CHECK: ldr [[GLOBALADDR:x[0-9]+]], {{\[}}[[GLOBALOFF]]]
; w0 is %arg
; CHECK-NEXT: sub w[[OFFSETREGNUM:[0-9]+]], w0, #1
; w1 is %var truncated
; CHECK-NEXT: strh w1, {{\[}}[[GLOBALADDR]], w[[OFFSETREGNUM]], sxtw #1]
; CHECK-NEXT: ret
bb:
  %.pre37 = load i16*, i16** @zptr16, align 8
  %dec = add nsw i32 %arg, -1
  %idxprom8 = sext i32 %dec to i64
  %arrayidx9 = getelementptr inbounds i16, i16* %.pre37, i64 %idxprom8
  %tmp = trunc i64 %var to i16
  store i16 %tmp, i16* %arrayidx9, align 4
  ret void
}

define void @fct8(i32 %arg, i64 %var) {
; CHECK: fct8
; CHECK: adrp [[GLOBALPAGE:x[0-9]+]], _zptr8@GOTPAGE
; CHECK: ldr [[GLOBALOFF:x[0-9]+]], {{\[}}[[GLOBALPAGE]], _zptr8@GOTPAGEOFF]
; CHECK: ldr [[BASEADDR:x[0-9]+]], {{\[}}[[GLOBALOFF]]]
; w0 is %arg
; CHECK-NEXT: add [[ADDR:x[0-9]+]], [[BASEADDR]], w0, sxtw
; w1 is %var truncated
; CHECK-NEXT: sturb w1, {{\[}}[[ADDR]], #-1]
; CHECK-NEXT: ret
bb:
  %.pre37 = load i8*, i8** @zptr8, align 8
  %dec = add nsw i32 %arg, -1
  %idxprom8 = sext i32 %dec to i64
  %arrayidx9 = getelementptr inbounds i8, i8* %.pre37, i64 %idxprom8
  %tmp = trunc i64 %var to i8
  store i8 %tmp, i8* %arrayidx9, align 4
  ret void
}
