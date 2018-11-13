; RUN: llc -o - %s -mtriple=aarch64-windows -verify-machineinstrs | FileCheck %s
; RUN: llc -o %t -filetype=obj %s -mtriple=aarch64-windows
; RUN: llvm-readobj -unwind %t | FileCheck %s -check-prefix=UNWIND

; We test the following
; 1) That the unwind help object is created and that its offset from the stack
;    pointer on entry is patched into the table fed to __CxxFrameHandler3
; 2) That the stack update for the catch funclet only includes the callee saved
;    registers
; 3) That the locals are accessed using the frame pointer in both the funclet
;    and the parent function.

; The following checks that the unwind help object has -2 stored into it at
; fp - 400 - 256 = fp - 656, which is on-entry sp - 48 + 32 - 656 =
; on-entry sp - 672.  We check this offset in the table later on.

; CHECK-LABEL: "?func@@YAHXZ":
; CHECK:       str     x28, [sp, #-48]!
; CHECK:       str     x21, [sp, #8]
; CHECK:       stp     x19, x20, [sp, #16]
; CHECK:       stp     x29, x30, [sp, #32]
; CHECK:       add     x29, sp, #32
; CHECK:       sub     sp, sp, #624
; CHECK:       mov     x19, sp
; CHECK:       orr     x1, xzr, #0xfffffffffffffffe
; CHECK:       stur    x1, [x19]

; Now check that x is stored at fp - 20.  We check that this is the same
; location accessed from the funclet to retrieve x.
; CHECK:       orr     w8, wzr, #0x1
; CHECK:       stur    w8, [x29, [[X_OFFSET:#-[1-9][0-9]+]]

; Check the offset off the frame pointer at which B is located.
; Check the same offset is used to pass the address of B to init2 in the
; funclet.
; CHECK:       sub     x0, x29, [[B_OFFSET:#[1-9][0-9]+]]
; CHECK:       bl      "?init@@YAXPEAH@Z"

; This is the label for the throw that is encoded in the ip2state.
; We are inside the try block, where we make a call to func2
; CHECK-LABEL: .Ltmp0:
; CHECK:       bl      "?func2@@YAHXZ

; CHECK:        [[CATCHRETDEST:.LBB0_[0-9]+]]:      ; %catchret.dest

; Check the catch funclet.
; CHECK-LABEL: "?catch$2@?0??func@@YAHXZ@4HA":

; Check that the stack space is allocated only for the callee saved registers.
; CHECK:       str     x28, [sp, #-48]!
; CHECK:       str     x21, [sp, #8]
; CHECK:       stp     x19, x20, [sp, #16]
; CHECK:       stp     x29, x30, [sp, #32]
; CHECK:       add     x20, x19, #12

; Check that there are no further stack updates.
; CHECK-NOT:   sub     sp, sp

; Check that the stack address passed to init2 is off the frame pointer, and
; that it matches the address of B in the parent function.
; CHECK:       sub     x0, x29, [[B_OFFSET]]
; CHECK:       bl      "?init2@@YAXPEAH@Z"

; Check that are storing x back to the same location off the frame pointer as in
; the parent function.
; CHECK:       stur    w8, [x29, [[X_OFFSET]]]

; Check that the funclet branches back to the catchret destination
; CHECK:       adrp    x0, .LBB0_3
; CHECK-NEXT:  add     x0, x0, [[CATCHRETDEST]]


; Now check that the offset of the unwind help object from the stack pointer on
; entry to func is encoded in cppxdata that is passed to __CxxFrameHandler3.  As
; computed above, this comes to -672.
; CHECK-LABEL:        "$cppxdata$?func@@YAHXZ":
; CHECK-NEXT:         .word   429065506               ; MagicNumber
; CHECK-NEXT:         .word   2                       ; MaxState
; CHECK-NEXT:         .word   ("$stateUnwindMap$?func@@YAHXZ")@IMGREL ; UnwindMap
; CHECK-NEXT:         .word   1                       ; NumTryBlocks
; CHECK-NEXT:         .word   ("$tryMap$?func@@YAHXZ")@IMGREL ; TryBlockMap
; CHECK-NEXT:         .word   4                       ; IPMapEntries
; CHECK-NEXT:         .word   ("$ip2state$?func@@YAHXZ")@IMGREL ; IPToStateXData
; CHECK-NEXT:         .word   -672                    ; UnwindHelp

; UNWIND: Function: ?func@@YAHXZ (0x0)
; UNWIND: Prologue [
; UNWIND-NEXT: ; nop
; UNWIND-NEXT: ; sub sp, #624
; UNWIND-NEXT: ; add fp, sp, #32
; UNWIND-NEXT: ; stp x29, x30, [sp, #32]
; UNWIND-NEXT: ; stp x19, x20, [sp, #16]
; UNWIND-NEXT: ; str x21, [sp, #8]
; UNWIND-NEXT: ; str x28, [sp, #48]!
; UNWIND-NEXT: ; end
; UNWIND: Function: ?catch$2@?0??func@@YAHXZ@4HA
; UNWIND: Prologue [
; UNWIND-NEXT: ; stp x29, x30, [sp, #32]
; UNWIND-NEXT: ; stp x19, x20, [sp, #16]
; UNWIND-NEXT: ; str x21, [sp, #8]
; UNWIND-NEXT: ; str x28, [sp, #48]!
; UNWIND-NEXT: ; end

target datalayout = "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-windows-msvc19.11.0"

%rtti.TypeDescriptor2 = type { i8**, i8*, [3 x i8] }
%eh.CatchableType = type { i32, i32, i32, i32, i32, i32, i32 }
%eh.CatchableTypeArray.1 = type { i32, [1 x i32] }
%eh.ThrowInfo = type { i32, i32, i32, i32 }

$"??_R0H@8" = comdat any

$"_CT??_R0H@84" = comdat any

$_CTA1H = comdat any

$_TI1H = comdat any

@"??_7type_info@@6B@" = external constant i8*
@"??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { i8** @"??_7type_info@@6B@", i8* null, [3 x i8] c".H\00" }, comdat
@__ImageBase = external dso_local constant i8
@"_CT??_R0H@84" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 1, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor2* @"??_R0H@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 4, i32 0 }, section ".xdata", comdat
@_CTA1H = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.1 { i32 1, [1 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%eh.CatchableType* @"_CT??_R0H@84" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32)] }, section ".xdata", comdat
@_TI1H = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%eh.CatchableTypeArray.1* @_CTA1H to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, section ".xdata", comdat

; Function Attrs: noinline optnone
define dso_local i32 @"?func@@YAHXZ"() #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %B = alloca [50 x i32], align 4
  %x = alloca i32, align 4
  %tmp = alloca i32, align 4
  %i = alloca i32, align 4
  %C = alloca [100 x i32], align 4
  store i32 1, i32* %x, align 4
  %arraydecay = getelementptr inbounds [50 x i32], [50 x i32]* %B, i32 0, i32 0
  call void @"?init@@YAXPEAH@Z"(i32* %arraydecay)
  %call = invoke i32 @"?func2@@YAHXZ"()
          to label %invoke.cont unwind label %catch.dispatch

invoke.cont:                                      ; preds = %entry
  store i32 %call, i32* %tmp, align 4
  %0 = bitcast i32* %tmp to i8*
  invoke void @_CxxThrowException(i8* %0, %eh.ThrowInfo* @_TI1H) #2
          to label %unreachable unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %invoke.cont, %entry
  %1 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %2 = catchpad within %1 [%rtti.TypeDescriptor2* @"??_R0H@8", i32 0, i32* %i]
  %arraydecay1 = getelementptr inbounds [100 x i32], [100 x i32]* %C, i32 0, i32 0
  call void @"?init@@YAXPEAH@Z"(i32* %arraydecay1) [ "funclet"(token %2) ]
  %arraydecay2 = getelementptr inbounds [50 x i32], [50 x i32]* %B, i32 0, i32 0
  call void @"?init2@@YAXPEAH@Z"(i32* %arraydecay2) [ "funclet"(token %2) ]
  %3 = load i32, i32* %i, align 4
  %idxprom = sext i32 %3 to i64
  %arrayidx = getelementptr inbounds [50 x i32], [50 x i32]* %B, i64 0, i64 %idxprom
  %4 = load i32, i32* %arrayidx, align 4
  %5 = load i32, i32* %i, align 4
  %idxprom3 = sext i32 %5 to i64
  %arrayidx4 = getelementptr inbounds [100 x i32], [100 x i32]* %C, i64 0, i64 %idxprom3
  %6 = load i32, i32* %arrayidx4, align 4
  %add = add nsw i32 %4, %6
  %7 = load i32, i32* %i, align 4
  %8 = load i32, i32* %i, align 4
  %mul = mul nsw i32 %7, %8
  %add5 = add nsw i32 %add, %mul
  store i32 %add5, i32* %x, align 4
  catchret from %2 to label %catchret.dest

catchret.dest:                                    ; preds = %catch
  br label %try.cont

try.cont:                                         ; preds = %catchret.dest
  %arrayidx6 = getelementptr inbounds [50 x i32], [50 x i32]* %B, i64 0, i64 2
  %9 = load i32, i32* %arrayidx6, align 4
  %10 = load i32, i32* %x, align 4
  %add7 = add nsw i32 %9, %10
  ret i32 %add7

unreachable:                                      ; preds = %invoke.cont
  unreachable
}

declare dso_local void @"?init@@YAXPEAH@Z"(i32*)

declare dso_local i32 @"?func2@@YAHXZ"()

declare dso_local i32 @__CxxFrameHandler3(...)

declare dllimport void @_CxxThrowException(i8*, %eh.ThrowInfo*)

declare dso_local void @"?init2@@YAXPEAH@Z"(i32*)

attributes #0 = { noinline optnone }
attributes #2 = { noreturn }
