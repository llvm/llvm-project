; RUN: llc -mtriple powerpc64-ibm-aix -verify-machineinstrs < %s | \
; RUN:  FileCheck %s

; RUN: llc -mtriple powerpc64-ibm-aix -fast-isel -verify-machineinstrs < %s | \
; RUN:  FileCheck %s

; Function Attrs: nounwind
declare i32 @func1() #0

declare i32 @__xlcxx_personality_v1(...)

; Function Attrs: mustprogress noinline optnone
define linkonce_odr void @func2() #1 align 2 personality i8* bitcast (i32 (...)* @__xlcxx_personality_v1 to i8*) {
entry:
  %0 = alloca i8*, align 8
  %1 = alloca i32, align 4
  br label %2

2:                                                ; preds = %3, %entry
  br i1 false, label %3, label %8

3:                                                ; preds = %2
  %4 = invoke i32 @func1()
          to label %2 unwind label %lpad

lpad:                                                ; preds = %3
  %5 = landingpad { i8*, i32 }
          cleanup
  %6 = extractvalue { i8*, i32 } %5, 0
  store i8* %6, i8** %0, align 8
  %7 = extractvalue { i8*, i32 } %5, 1
  store i32 %7, i32* %1, align 4
  br label %eh.resume

8:                                               ; preds = 2%
  ret void

eh.resume:                                               ; preds = %lpad
  %9 = load i8*, i8** %0, align 8
  %10 = load i32, i32* %1, align 4
  %11 = insertvalue { i8*, i32 } undef, i8* %9, 0
  %12 = insertvalue { i8*, i32 } %11, i32 %10, 1
  resume { i8*, i32 } %12
}

attributes #0 = { nounwind }
attributes #1 = { mustprogress noinline optnone }

; CHECK: __ehinfo.0:
; CHECK: .tc __ehinfo.0[TC],__ehinfo.0
