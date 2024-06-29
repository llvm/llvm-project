; RUN: llc -mtriple=arm-netbsd-eabi -o - -filetype=asm -simplifycfg-require-and-preserve-domtree=1 %s | \
; RUN: FileCheck %s
; RUN: llc -mtriple=arm-netbsd-eabi -o - -filetype=asm -simplifycfg-require-and-preserve-domtree=1 %s \
; RUN: -relocation-model=pic | FileCheck -check-prefix=CHECK-PIC %s

; ModuleID = 'test.cc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:64-n32-S64"
target triple = "armv5e--netbsd-eabi"

%struct.exception = type { i8 }

@_ZTVN10__cxxabiv117__class_type_infoE = external global ptr
@_ZTS9exception = linkonce_odr constant [11 x i8] c"9exception\00"
@_ZTI9exception = linkonce_odr unnamed_addr constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i32 2), ptr @_ZTS9exception }

define void @f() uwtable personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr
  %2 = alloca i32
  %e = alloca ptr, align 4
  invoke void @g()
          to label %3 unwind label %4

  br label %16

  %5 = landingpad { ptr, i32 }
          catch ptr @_ZTI9exception
  %6 = extractvalue { ptr, i32 } %5, 0
  store ptr %6, ptr %1
  %7 = extractvalue { ptr, i32 } %5, 1
  store i32 %7, ptr %2
  br label %8

  %9 = load i32, ptr %2
  %10 = call i32 @llvm.eh.typeid.for(ptr @_ZTI9exception) nounwind
  %11 = icmp eq i32 %9, %10
  br i1 %11, label %12, label %17

  %13 = load ptr, ptr %1
  %14 = call ptr @__cxa_begin_catch(ptr %13) #3
  %15 = bitcast ptr %14 to ptr
  store ptr %15, ptr %e
  call void @__cxa_end_catch()
  br label %16

  ret void

  %18 = load ptr, ptr %1
  %19 = load i32, ptr %2
  %20 = insertvalue { ptr, i32 } undef, ptr %18, 0
  %21 = insertvalue { ptr, i32 } %20, i32 %19, 1
  resume { ptr, i32 } %21
}

declare void @g()

declare i32 @__gxx_personality_v0(...)

declare i32 @llvm.eh.typeid.for(ptr) nounwind readnone

declare ptr @__cxa_begin_catch(ptr)

declare void @__cxa_end_catch()

; CHECK: .cfi_personality 0,
; CHECK: .cfi_lsda 0,
; CHECK: @TType Encoding = absptr
; CHECK: @ Call site Encoding = uleb128
; CHECK-PIC: .cfi_personality 155,
; CHECK-PIC: .cfi_lsda 27,
; CHECK-PIC: @TType Encoding = indirect pcrel sdata4
; CHECK-PIC: @ Call site Encoding = uleb128
