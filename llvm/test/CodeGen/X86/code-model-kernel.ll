; RUN: llc -mtriple=x86_64-pc-linux-gnu -code-model=kernel %s -o - | FileCheck %s
; CHECK-LABEL: main
; CHECK: .cfi_startproc
; CHECK: .cfi_personality 0, __gxx_personality_v0
; CHECK: .cfi_lsda 0, [[EXCEPTION_LABEL:.L[^ ]*]]
; CHECK: [[EXCEPTION_LABEL]]:
; CHECK: .byte	0                       # @TType Encoding = absptr
; CHECK: .quad	_ZTIi

@_ZTIi = external constant i8*

; Function Attrs: noinline norecurse optnone uwtable
define i32 @main() #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %1 = alloca i32, align 4
  %2 = alloca i8*
  %3 = alloca i32
  %4 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  %5 = call i8* @__cxa_allocate_exception(i64 4) #2
  %6 = bitcast i8* %5 to i32*
  store i32 20, i32* %6, align 16
  invoke void @__cxa_throw(i8* %5, i8* bitcast (i8** @_ZTIi to i8*), i8* null) #3
          to label %26 unwind label %7

; <label>:7:                                      ; preds = %0
  %8 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %9 = extractvalue { i8*, i32 } %8, 0
  store i8* %9, i8** %2, align 8
  %10 = extractvalue { i8*, i32 } %8, 1
  store i32 %10, i32* %3, align 4
  br label %11

; <label>:11:                                     ; preds = %7
  %12 = load i32, i32* %3, align 4
  %13 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #2
  %14 = icmp eq i32 %12, %13
  br i1 %14, label %15, label %21

; <label>:15:                                     ; preds = %11
  %16 = load i8*, i8** %2, align 8
  %17 = call i8* @__cxa_begin_catch(i8* %16) #2
  %18 = bitcast i8* %17 to i32*
  %19 = load i32, i32* %18, align 4
  store i32 %19, i32* %4, align 4
  call void @__cxa_end_catch() #2
  br label %20

; <label>:20:                                     ; preds = %15
  ret i32 0

; <label>:21:                                     ; preds = %11
  %22 = load i8*, i8** %2, align 8
  %23 = load i32, i32* %3, align 4
  %24 = insertvalue { i8*, i32 } undef, i8* %22, 0
  %25 = insertvalue { i8*, i32 } %24, i32 %23, 1
  resume { i8*, i32 } %25

; <label>:26:                                     ; preds = %0
  unreachable
}

declare i8* @__cxa_allocate_exception(i64)

declare void @__cxa_throw(i8*, i8*, i8*)

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #1

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()

attributes #0 = { noinline norecurse optnone uwtable }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
attributes #3 = { noreturn }
