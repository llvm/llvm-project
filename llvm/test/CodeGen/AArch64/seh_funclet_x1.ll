; RUN: llc -o - %s -verify-machineinstrs -mtriple=aarch64-windows | FileCheck %s

; Windows runtime passes the establisher frame as the second argument to the
; termination handler.  Check that we copy it into fp.

; CHECK:      ?dtor$3@?0?main@4HA":
; CHECK:      .seh_proc "?dtor$3@?0?main@4HA"
; CHECK:      stp     x29, x30, [sp, #-16]!   // 16-byte Folded Spill
; CHECK-NEXT: .seh_save_fplr_x 16
; CHECK-NEXT: .seh_endprologue
; CHECK-NEXT: mov     x29, x1

target datalayout = "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-windows-msvc19.15.26732"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*) {
entry:
  %retval = alloca i32, align 4
  %Counter = alloca i32, align 4
  %__exception_code = alloca i32, align 4
  call void (...) @llvm.localescape(i32* %Counter)
  store i32 0, i32* %retval, align 4
  store i32 0, i32* %Counter, align 4
  %call = invoke i32 bitcast (i32 (...)* @RaiseStatus to i32 (i32)*)(i32 -1073741675) #3
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  %0 = call i8* @llvm.localaddress()
  invoke void @"?fin$0@0@main@@"(i8 0, i8* %0) #3
          to label %invoke.cont1 unwind label %catch.dispatch

invoke.cont1:                                     ; preds = %invoke.cont
  br label %__try.cont

ehcleanup:                                        ; preds = %entry
  %1 = cleanuppad within none []
  %2 = call i8* @llvm.localaddress()
  invoke void @"?fin$0@0@main@@"(i8 1, i8* %2) #3 [ "funclet"(token %1) ]
          to label %invoke.cont2 unwind label %catch.dispatch

invoke.cont2:                                     ; preds = %ehcleanup
  cleanupret from %1 unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %invoke.cont2, %ehcleanup, %invoke.cont
  %3 = catchswitch within none [label %__except] unwind to caller

__except:                                         ; preds = %catch.dispatch
  %4 = catchpad within %3 [i8* null]
  catchret from %4 to label %__except3

__except3:                                        ; preds = %__except
  %5 = call i32 @llvm.eh.exceptioncode(token %4)
  store i32 %5, i32* %__exception_code, align 4
  %6 = load i32, i32* %Counter, align 4
  %add = add nsw i32 %6, 5
  store i32 %add, i32* %Counter, align 4
  br label %__try.cont

__try.cont:                                       ; preds = %__except3, %invoke.cont1
  %7 = load i32, i32* %retval, align 4
  ret i32 %7
}

define internal void @"?fin$0@0@main@@"(i8 %abnormal_termination, i8* %frame_pointer) {
entry:
  %frame_pointer.addr = alloca i8*, align 8
  %abnormal_termination.addr = alloca i8, align 1
  %0 = call i8* @llvm.localrecover(i8* bitcast (i32 ()* @main to i8*), i8* %frame_pointer, i32 0)
  %Counter = bitcast i8* %0 to i32*
  store i8* %frame_pointer, i8** %frame_pointer.addr, align 8
  store i8 %abnormal_termination, i8* %abnormal_termination.addr, align 1
  store i32 3, i32* %Counter, align 4
  call void @"?fin$1@0@main@@"(i8 0, i8* %frame_pointer)
  %1 = load i32, i32* %Counter, align 4
  %add = add nsw i32 %1, 2
  store i32 %add, i32* %Counter, align 4
  %call = call i32 bitcast (i32 (...)* @RaiseStatus to i32 (i32)*)(i32 -1073741675)
  ret void
}

; Function Attrs: nounwind readnone
declare i8* @llvm.localrecover(i8*, i8*, i32) #1

define internal void @"?fin$1@0@main@@"(i8 %abnormal_termination, i8* %frame_pointer) {
entry:
  %frame_pointer.addr = alloca i8*, align 8
  %abnormal_termination.addr = alloca i8, align 1
  %0 = call i8* @llvm.localrecover(i8* bitcast (i32 ()* @main to i8*), i8* %frame_pointer, i32 0)
  %Counter = bitcast i8* %0 to i32*
  store i8* %frame_pointer, i8** %frame_pointer.addr, align 8
  store i8 %abnormal_termination, i8* %abnormal_termination.addr, align 1
  %1 = load i32, i32* %Counter, align 4
  %cmp = icmp eq i32 %1, 3
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %2 = load i32, i32* %Counter, align 4
  %add = add nsw i32 %2, 1
  store i32 %add, i32* %Counter, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

declare dso_local i32 @RaiseStatus(...)

declare dso_local i32 @__C_specific_handler(...)

; Function Attrs: nounwind readnone
declare i8* @llvm.localaddress() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.exceptioncode(token) #1

; Function Attrs: nounwind
declare void @llvm.localescape(...) #2

attributes #0 = { noinline nounwind optnone uwtable }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
attributes #3 = { noinline }
