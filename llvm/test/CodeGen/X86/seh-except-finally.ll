; RUN: llc < %s | FileCheck %s

; Test case based on this source:
; int puts(const char*);
; __declspec(noinline) void crash() {
;   *(volatile int*)0 = 42;
; }
; int filt();
; void use_both() {
;   __try {
;     __try {
;       crash();
;     } __finally {
;       puts("__finally");
;     }
;   } __except (filt()) {
;     puts("__except");
;   }
; }

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

$"\01??_C@_09KJEHOMHG@__finally?$AA@" = comdat any

$"\01??_C@_08MLCMLGHM@__except?$AA@" = comdat any

@"\01??_C@_09KJEHOMHG@__finally?$AA@" = linkonce_odr unnamed_addr constant [10 x i8] c"__finally\00", comdat, align 1
@"\01??_C@_08MLCMLGHM@__except?$AA@" = linkonce_odr unnamed_addr constant [9 x i8] c"__except\00", comdat, align 1

declare void @crash()

declare i32 @filt()

; Function Attrs: nounwind uwtable
define void @use_both() #1 personality ptr @__C_specific_handler {
entry:
  %exn.slot = alloca ptr
  %ehselector.slot = alloca i32
  invoke void @crash() #5
          to label %invoke.cont unwind label %__finally

invoke.cont:                                      ; preds = %entry
  %0 = call ptr @llvm.localaddress()
  invoke void @"\01?fin$0@0@use_both@@"(i1 zeroext false, ptr %0) #5
          to label %invoke.cont2 unwind label %catch.dispatch

invoke.cont2:                                     ; preds = %invoke.cont
  br label %__try.cont

__finally:                                             ; preds = %entry
  %cleanuppad = cleanuppad within none []
  %locals = call ptr @llvm.localaddress()
  invoke void @"\01?fin$0@0@use_both@@"(i1 zeroext true, ptr %locals) #5 [ "funclet"(token %cleanuppad) ]
          to label %invoke.cont3 unwind label %catch.dispatch

invoke.cont3:                                     ; preds = %__finally
  cleanupret from %cleanuppad unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %invoke.cont3, %lpad1
  %cs1 = catchswitch within none [label %__except] unwind to caller

__except:                                         ; preds = %catch.dispatch
  %catchpad = catchpad within %cs1 [ptr @"\01?filt$0@0@use_both@@"]
  %call = call i32 @puts(ptr @"\01??_C@_08MLCMLGHM@__except?$AA@") [ "funclet"(token %catchpad) ]
  catchret from %catchpad to label %__try.cont

__try.cont:                                       ; preds = %__except, %invoke.cont2
  ret void
}

; CHECK-LABEL: use_both:
; CHECK: .Ltmp0
; CHECK: callq crash
; CHECK: .Ltmp1
; CHECK: .Ltmp4
; CHECK: callq "?fin$0@0@use_both@@"
; CHECK: .Ltmp5
; CHECK: retq
;
; CHECK: .seh_handlerdata
; CHECK-NEXT: .Luse_both$parent_frame_offset
; CHECK-NEXT: .long (.Llsda_end0-.Llsda_begin0)/16
; CHECK-NEXT: .Llsda_begin0:
; CHECK-NEXT: .long .Ltmp0@IMGREL
; CHECK-NEXT: .long .Ltmp1@IMGREL+1
; CHECK-NEXT: .long "?dtor$2@?0?use_both@4HA"@IMGREL
; CHECK-NEXT: .long 0
; CHECK-NEXT: .long .Ltmp0@IMGREL
; CHECK-NEXT: .long .Ltmp1@IMGREL+1
; CHECK-NEXT: .long "?filt$0@0@use_both@@"@IMGREL
; CHECK-NEXT: .long .LBB0_{{[0-9]+}}@IMGREL
; CHECK-NEXT: .long .Ltmp4@IMGREL
; CHECK-NEXT: .long .Ltmp5@IMGREL+1
; CHECK-NEXT: .long "?filt$0@0@use_both@@"@IMGREL
; CHECK-NEXT: .long .LBB0_{{[0-9]+}}@IMGREL
; CHECK-NEXT: .Llsda_end0:

; Function Attrs: noinline nounwind
define internal i32 @"\01?filt$0@0@use_both@@"(ptr %exception_pointers, ptr %frame_pointer) #2 {
entry:
  %frame_pointer.addr = alloca ptr, align 8
  %exception_pointers.addr = alloca ptr, align 8
  %exn.slot = alloca ptr
  store ptr %frame_pointer, ptr %frame_pointer.addr, align 8
  store ptr %exception_pointers, ptr %exception_pointers.addr, align 8
  %0 = load ptr, ptr %exception_pointers.addr
  %1 = getelementptr inbounds { ptr, ptr }, ptr %0, i32 0, i32 0
  %2 = load ptr, ptr %1
  %3 = load i32, ptr %2
  %4 = zext i32 %3 to i64
  %5 = inttoptr i64 %4 to ptr
  store ptr %5, ptr %exn.slot
  %call = call i32 @filt()
  ret i32 %call
}

define internal void @"\01?fin$0@0@use_both@@"(i1 zeroext %abnormal_termination, ptr %frame_pointer) #3 {
entry:
  %frame_pointer.addr = alloca ptr, align 8
  %abnormal_termination.addr = alloca i8, align 1
  store ptr %frame_pointer, ptr %frame_pointer.addr, align 8
  %frombool = zext i1 %abnormal_termination to i8
  store i8 %frombool, ptr %abnormal_termination.addr, align 1
  %call = call i32 @puts(ptr @"\01??_C@_09KJEHOMHG@__finally?$AA@")
  ret void
}

declare i32 @puts(ptr) #3

declare i32 @__C_specific_handler(...)

; Function Attrs: nounwind readnone
declare ptr @llvm.localaddress() #4

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(ptr) #4

attributes #0 = { noinline nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { noinline nounwind "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind readnone }
attributes #5 = { noinline }
attributes #6 = { nounwind }
