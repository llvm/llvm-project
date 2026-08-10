; RUN: llc  -mtriple=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

;16: main:
;16-NEXT: [[TMP:.*]]:
;16-NEXT: $func_begin0 = [[TMP]]
;16-NEXT: .cfi_startproc
;16-NEXT: .cfi_personality
@.str = private unnamed_addr constant [7 x i8] c"hello\0A\00", align 1
@_ZTIi = external constant ptr
@.str1 = private unnamed_addr constant [15 x i8] c"exception %i \0A\00", align 1

define i32 @main() personality ptr @__gxx_personality_v0 {
entry:
  %retval = alloca i32, align 4
  %exn.slot = alloca ptr
  %ehselector.slot = alloca i32
  %e = alloca i32, align 4
  store i32 0, ptr %retval
  %call = call i32 (ptr, ...) @printf(ptr @.str)
  %exception = call ptr @__cxa_allocate_exception(i32 4) nounwind
  store i32 20, ptr %exception
  invoke void @__cxa_throw(ptr %exception, ptr @_ZTIi, ptr null) noreturn
          to label %unreachable unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %1 = extractvalue { ptr, i32 } %0, 0
  store ptr %1, ptr %exn.slot
  %2 = extractvalue { ptr, i32 } %0, 1
  store i32 %2, ptr %ehselector.slot
  br label %catch.dispatch

catch.dispatch:                                   ; preds = %lpad
  %sel = load i32, ptr %ehselector.slot
  %3 = call i32 @llvm.eh.typeid.for(ptr @_ZTIi) nounwind
  %matches = icmp eq i32 %sel, %3
  br i1 %matches, label %catch, label %eh.resume

catch:                                            ; preds = %catch.dispatch
  %exn = load ptr, ptr %exn.slot
  %4 = call ptr @__cxa_begin_catch(ptr %exn) nounwind
  %exn.scalar = load i32, ptr %4
  store i32 %exn.scalar, ptr %e, align 4
  %5 = load i32, ptr %e, align 4
  %call2 = invoke i32 (ptr, ...) @printf(ptr @.str1, i32 %5)
          to label %invoke.cont unwind label %lpad1

invoke.cont:                                      ; preds = %catch
  call void @__cxa_end_catch() nounwind
  br label %try.cont

try.cont:                                         ; preds = %invoke.cont
  ret i32 0

lpad1:                                            ; preds = %catch
  %6 = landingpad { ptr, i32 }
          cleanup
  %7 = extractvalue { ptr, i32 } %6, 0
  store ptr %7, ptr %exn.slot
  %8 = extractvalue { ptr, i32 } %6, 1
  store i32 %8, ptr %ehselector.slot
  call void @__cxa_end_catch() nounwind
  br label %eh.resume

eh.resume:                                        ; preds = %lpad1, %catch.dispatch
  %exn3 = load ptr, ptr %exn.slot
  %sel4 = load i32, ptr %ehselector.slot
  %lpad.val = insertvalue { ptr, i32 } undef, ptr %exn3, 0
  %lpad.val5 = insertvalue { ptr, i32 } %lpad.val, i32 %sel4, 1
  resume { ptr, i32 } %lpad.val5

unreachable:                                      ; preds = %entry
  unreachable
}

declare i32 @printf(ptr, ...)

declare ptr @__cxa_allocate_exception(i32)

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_throw(ptr, ptr, ptr)

declare i32 @llvm.eh.typeid.for(ptr) nounwind readnone

declare ptr @__cxa_begin_catch(ptr)

declare void @__cxa_end_catch()
