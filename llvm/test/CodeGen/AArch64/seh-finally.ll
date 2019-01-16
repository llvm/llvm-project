; RUN: llc -mtriple arm64-windows -o - %s | FileCheck %s

; Function Attrs: noinline optnone uwtable
define dso_local i32 @foo() {
entry:
; CHECK-LABEL: foo
; CHECK: orr     w8, wzr, #0x1
; CHECK: mov     w0, wzr
; CHECK: mov     x1, x29
; CHECK: .set .Lfoo$frame_escape_0, -4
; CHECK: stur    w8, [x29, #-4]
; CHECK: bl      "?fin$0@0@foo@@"
; CHECK: ldur    w0, [x29, #-4]

  %count = alloca i32, align 4
  call void (...) @llvm.localescape(i32* %count)
  store i32 0, i32* %count, align 4
  %0 = load i32, i32* %count, align 4
  %add = add nsw i32 %0, 1
  store i32 %add, i32* %count, align 4
  %1 = call i8* @llvm.localaddress()
  call void @"?fin$0@0@foo@@"(i8 0, i8* %1)
  %2 = load i32, i32* %count, align 4
  ret i32 %2
}

define internal void @"?fin$0@0@foo@@"(i8 %abnormal_termination, i8* %frame_pointer) {
entry:
; CHECK-LABEL: @"?fin$0@0@foo@@"
; CHECK: sub     sp, sp, #16
; CHECK: str     x1, [sp, #8]
; CHECK: strb    w0, [sp, #7]
; CHECK: movz    x8, #:abs_g1_s:.Lfoo$frame_escape_0
; CHECK: movk    x8, #:abs_g0_nc:.Lfoo$frame_escape_0
; CHECK: add     x8, x1, x8
; CHECK: ldr     w9, [x8]
; CHECK: add     w9, w9, #1
; CHECK: str     w9, [x8]

  %frame_pointer.addr = alloca i8*, align 8
  %abnormal_termination.addr = alloca i8, align 1
  %0 = call i8* @llvm.localrecover(i8* bitcast (i32 ()* @foo to i8*), i8* %frame_pointer, i32 0)
  %count = bitcast i8* %0 to i32*
  store i8* %frame_pointer, i8** %frame_pointer.addr, align 8
  store i8 %abnormal_termination, i8* %abnormal_termination.addr, align 1
  %1 = zext i8 %abnormal_termination to i32
  %cmp = icmp eq i32 %1, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %2 = load i32, i32* %count, align 4
  %add = add nsw i32 %2, 1
  store i32 %add, i32* %count, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; Function Attrs: nounwind readnone
declare i8* @llvm.localrecover(i8*, i8*, i32)

; Function Attrs: nounwind readnone
declare i8* @llvm.localaddress()

; Function Attrs: nounwind
declare void @llvm.localescape(...)
