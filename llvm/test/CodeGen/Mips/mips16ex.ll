; RUN: llc  -march=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

;16: main:
;16-NEXT: [[TMP:.*]]:
;16-NEXT: .set $func_begin0, ([[TMP]])
;16-NEXT: .cfi_startproc
;16-NEXT: .cfi_personality
@.str = private unnamed_addr constant [7 x i8] c"hello\0A\00", align 1
@_ZTIi = external constant i8*
@.str1 = private unnamed_addr constant [15 x i8] c"exception %i \0A\00", align 1

define i32 @main() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %retval = alloca i32, align 4
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  %e = alloca i32, align 4
  store i32 0, i32* %retval
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str, i32 0, i32 0))
  %exception = call i8* @__cxa_allocate_exception(i32 4) nounwind
  %0 = bitcast i8* %exception to i32*
  store i32 20, i32* %0
  invoke void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null) noreturn
          to label %unreachable unwind label %lpad

lpad:                                             ; preds = %entry
  %1 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %2 = extractvalue { i8*, i32 } %1, 0
  store i8* %2, i8** %exn.slot
  %3 = extractvalue { i8*, i32 } %1, 1
  store i32 %3, i32* %ehselector.slot
  br label %catch.dispatch

catch.dispatch:                                   ; preds = %lpad
  %sel = load i32, i32* %ehselector.slot
  %4 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) nounwind
  %matches = icmp eq i32 %sel, %4
  br i1 %matches, label %catch, label %eh.resume

catch:                                            ; preds = %catch.dispatch
  %exn = load i8*, i8** %exn.slot
  %5 = call i8* @__cxa_begin_catch(i8* %exn) nounwind
  %6 = bitcast i8* %5 to i32*
  %exn.scalar = load i32, i32* %6
  store i32 %exn.scalar, i32* %e, align 4
  %7 = load i32, i32* %e, align 4
  %call2 = invoke i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str1, i32 0, i32 0), i32 %7)
          to label %invoke.cont unwind label %lpad1

invoke.cont:                                      ; preds = %catch
  call void @__cxa_end_catch() nounwind
  br label %try.cont

try.cont:                                         ; preds = %invoke.cont
  ret i32 0

lpad1:                                            ; preds = %catch
  %8 = landingpad { i8*, i32 }
          cleanup
  %9 = extractvalue { i8*, i32 } %8, 0
  store i8* %9, i8** %exn.slot
  %10 = extractvalue { i8*, i32 } %8, 1
  store i32 %10, i32* %ehselector.slot
  call void @__cxa_end_catch() nounwind
  br label %eh.resume

eh.resume:                                        ; preds = %lpad1, %catch.dispatch
  %exn3 = load i8*, i8** %exn.slot
  %sel4 = load i32, i32* %ehselector.slot
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn3, 0
  %lpad.val5 = insertvalue { i8*, i32 } %lpad.val, i32 %sel4, 1
  resume { i8*, i32 } %lpad.val5

unreachable:                                      ; preds = %entry
  unreachable
}

declare i32 @printf(i8*, ...)

declare i8* @__cxa_allocate_exception(i32)

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_throw(i8*, i8*, i8*)

declare i32 @llvm.eh.typeid.for(i8*) nounwind readnone

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()
