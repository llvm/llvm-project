; RUN: llc -mtriple=x86_64-pc-linux-gnu -code-model=kernel -simplifycfg-require-and-preserve-domtree=1 %s -o - | FileCheck %s
; CHECK-LABEL: main
; CHECK: .cfi_startproc
; CHECK: .cfi_personality 0, __gxx_personality_v0
; CHECK: .cfi_lsda 0, [[EXCEPTION_LABEL:.L[^ ]*]]
; CHECK: [[EXCEPTION_LABEL]]:
; CHECK: .byte	0                       # @TType Encoding = absptr
; CHECK: .quad	_ZTIi

@_ZTIi = external constant ptr

; Function Attrs: noinline norecurse optnone uwtable
define i32 @main() #0 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca ptr
  %3 = alloca i32
  %4 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  %5 = call ptr @__cxa_allocate_exception(i64 4) #2
  %6 = bitcast ptr %5 to ptr
  store i32 20, ptr %6, align 16
  invoke void @__cxa_throw(ptr %5, ptr @_ZTIi, ptr null) #3
          to label %26 unwind label %7

; <label>:7:                                      ; preds = %0
  %8 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %9 = extractvalue { ptr, i32 } %8, 0
  store ptr %9, ptr %2, align 8
  %10 = extractvalue { ptr, i32 } %8, 1
  store i32 %10, ptr %3, align 4
  br label %11

; <label>:11:                                     ; preds = %7
  %12 = load i32, ptr %3, align 4
  %13 = call i32 @llvm.eh.typeid.for(ptr @_ZTIi) #2
  %14 = icmp eq i32 %12, %13
  br i1 %14, label %15, label %21

; <label>:15:                                     ; preds = %11
  %16 = load ptr, ptr %2, align 8
  %17 = call ptr @__cxa_begin_catch(ptr %16) #2
  %18 = bitcast ptr %17 to ptr
  %19 = load i32, ptr %18, align 4
  store i32 %19, ptr %4, align 4
  call void @__cxa_end_catch() #2
  br label %20

; <label>:20:                                     ; preds = %15
  ret i32 0

; <label>:21:                                     ; preds = %11
  %22 = load ptr, ptr %2, align 8
  %23 = load i32, ptr %3, align 4
  %24 = insertvalue { ptr, i32 } undef, ptr %22, 0
  %25 = insertvalue { ptr, i32 } %24, i32 %23, 1
  resume { ptr, i32 } %25

; <label>:26:                                     ; preds = %0
  unreachable
}

declare ptr @__cxa_allocate_exception(i64)

declare void @__cxa_throw(ptr, ptr, ptr)

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(ptr) #1

declare ptr @__cxa_begin_catch(ptr)

declare void @__cxa_end_catch()

attributes #0 = { noinline norecurse optnone uwtable }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
attributes #3 = { noreturn }
