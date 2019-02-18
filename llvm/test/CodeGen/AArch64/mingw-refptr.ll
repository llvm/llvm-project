; RUN: llc < %s -mtriple=aarch64-w64-mingw32 | FileCheck %s

@var = external local_unnamed_addr global i32, align 4
@dsolocalvar = external dso_local local_unnamed_addr global i32, align 4
@localvar = dso_local local_unnamed_addr global i32 0, align 4
@localcommon = common dso_local local_unnamed_addr global i32 0, align 4
@extvar = external dllimport local_unnamed_addr global i32, align 4

define dso_local i32 @getVar() {
; CHECK-LABEL: getVar:
; CHECK:    adrp x8, .refptr.var
; CHECK:    ldr  x8, [x8, .refptr.var]
; CHECK:    ldr  w0, [x8]
; CHECK:    ret
entry:
  %0 = load i32, i32* @var, align 4
  ret i32 %0
}

define dso_local i32 @getDsoLocalVar() {
; CHECK-LABEL: getDsoLocalVar:
; CHECK:    adrp x8, dsolocalvar
; CHECK:    ldr  w0, [x8, dsolocalvar]
; CHECK:    ret
entry:
  %0 = load i32, i32* @dsolocalvar, align 4
  ret i32 %0
}

define dso_local i32 @getLocalVar() {
; CHECK-LABEL: getLocalVar:
; CHECK:    adrp x8, localvar
; CHECK:    ldr  w0, [x8, localvar]
; CHECK:    ret
entry:
  %0 = load i32, i32* @localvar, align 4
  ret i32 %0
}

define dso_local i32 @getLocalCommon() {
; CHECK-LABEL: getLocalCommon:
; CHECK:    adrp x8, localcommon
; CHECK:    ldr  w0, [x8, localcommon]
; CHECK:    ret
entry:
  %0 = load i32, i32* @localcommon, align 4
  ret i32 %0
}

define dso_local i32 @getExtVar() {
; CHECK-LABEL: getExtVar:
; CHECK:    adrp x8, __imp_extvar
; CHECK:    ldr  x8, [x8, __imp_extvar]
; CHECK:    ldr  w0, [x8]
; CHECK:    ret
entry:
  %0 = load i32, i32* @extvar, align 4
  ret i32 %0
}

define dso_local void @callFunc() {
; CHECK-LABEL: callFunc:
; CHECK:    b otherFunc
entry:
  tail call void @otherFunc()
  ret void
}

declare dso_local void @otherFunc()

define dso_local void @sspFunc() #0 {
; CHECK-LABEL: sspFunc:
; CHECK:    adrp x8, .refptr.__stack_chk_guard
; CHECK:    ldr  x8, [x8, .refptr.__stack_chk_guard]
; CHECK:    ldr  x8, [x8]
entry:
  %c = alloca i8, align 1
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %c)
  call void @ptrUser(i8* nonnull %c)
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %c)
  ret void
}

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare dso_local void @ptrUser(i8*) local_unnamed_addr #2
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)

attributes #0 = { sspstrong }

; CHECK:        .section        .rdata$.refptr.__stack_chk_guard,"dr",discard,.refptr.__stack_chk_guard
; CHECK:        .globl  .refptr.__stack_chk_guard
; CHECK: .refptr.__stack_chk_guard:
; CHECK:        .xword  __stack_chk_guard
; CHECK:        .section        .rdata$.refptr.var,"dr",discard,.refptr.var
; CHECK:        .globl  .refptr.var
; CHECK: .refptr.var:
; CHECK:        .xword  var
