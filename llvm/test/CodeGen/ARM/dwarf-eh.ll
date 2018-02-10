; RUN: llc -mtriple=arm-netbsd-eabi -o - -filetype=asm %s | \
; RUN: FileCheck %s
; RUN: llc -mtriple=arm-netbsd-eabi -o - -filetype=asm %s \
; RUN: -relocation-model=pic | FileCheck -check-prefix=CHECK-PIC %s

; ModuleID = 'test.cc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:64-n32-S64"
target triple = "armv5e--netbsd-eabi"

%struct.exception = type { i8 }

@_ZTVN10__cxxabiv117__class_type_infoE = external global i8*
@_ZTS9exception = linkonce_odr constant [11 x i8] c"9exception\00"
@_ZTI9exception = linkonce_odr unnamed_addr constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i32 2) to i8*), i8* getelementptr inbounds ([11 x i8], [11 x i8]* @_ZTS9exception, i32 0, i32 0) }

define void @f() uwtable personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %1 = alloca i8*
  %2 = alloca i32
  %e = alloca %struct.exception*, align 4
  invoke void @g()
          to label %3 unwind label %4

  br label %16

  %5 = landingpad { i8*, i32 }
          catch i8* bitcast ({ i8*, i8* }* @_ZTI9exception to i8*)
  %6 = extractvalue { i8*, i32 } %5, 0
  store i8* %6, i8** %1
  %7 = extractvalue { i8*, i32 } %5, 1
  store i32 %7, i32* %2
  br label %8

  %9 = load i32, i32* %2
  %10 = call i32 @llvm.eh.typeid.for(i8* bitcast ({ i8*, i8* }* @_ZTI9exception to i8*)) nounwind
  %11 = icmp eq i32 %9, %10
  br i1 %11, label %12, label %17

  %13 = load i8*, i8** %1
  %14 = call i8* @__cxa_begin_catch(i8* %13) #3
  %15 = bitcast i8* %14 to %struct.exception*
  store %struct.exception* %15, %struct.exception** %e
  call void @__cxa_end_catch()
  br label %16

  ret void

  %18 = load i8*, i8** %1
  %19 = load i32, i32* %2
  %20 = insertvalue { i8*, i32 } undef, i8* %18, 0
  %21 = insertvalue { i8*, i32 } %20, i32 %19, 1
  resume { i8*, i32 } %21
}

declare void @g()

declare i32 @__gxx_personality_v0(...)

declare i32 @llvm.eh.typeid.for(i8*) nounwind readnone

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()

; CHECK: .cfi_personality 0,
; CHECK: .cfi_lsda 0,
; CHECK: @TType Encoding = absptr
; CHECK: @ Call site Encoding = uleb128
; CHECK-PIC: .cfi_personality 155,
; CHECK-PIC: .cfi_lsda 27,
; CHECK-PIC: @TType Encoding = indirect pcrel sdata4
; CHECK-PIC: @ Call site Encoding = uleb128
