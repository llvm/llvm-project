; Check if landing pads are kept in a separate eh section
; RUN: llc -simplifycfg-require-and-preserve-domtree=1 < %s -mtriple=i386-unknown-linux-gnu  -function-sections -basic-block-sections=all -unique-basic-block-section-names | FileCheck %s -check-prefix=LINUX-SECTIONS

@_ZTIb = external dso_local constant ptr
define i32 @_Z3foob(i1 zeroext %0) #0 personality ptr @__gxx_personality_v0 {
  %2 = alloca i32, align 4
  %3 = alloca i8, align 1
  %4 = alloca ptr
  %5 = alloca i32
  %6 = alloca i8, align 1
  %7 = zext i1 %0 to i8
  store i8 %7, ptr %3, align 1
  %8 = load i8, ptr %3, align 1
  %9 = trunc i8 %8 to i1
  br i1 %9, label %10, label %11

10:                                               ; preds = %1
  store i32 1, ptr %2, align 4
  br label %31

11:                                               ; preds = %1
  %12 = call ptr @__cxa_allocate_exception(i64 1) #2
  %13 = load i8, ptr %3, align 1
  %14 = trunc i8 %13 to i1
  %15 = zext i1 %14 to i8
  store i8 %15, ptr %12, align 16
  invoke void @__cxa_throw(ptr %12, ptr @_ZTIb, ptr null) #3
          to label %38 unwind label %16

16:                                               ; preds = %11
  %17 = landingpad { ptr, i32 }
          catch ptr @_ZTIb
  %18 = extractvalue { ptr, i32 } %17, 0
  store ptr %18, ptr %4, align 8
  %19 = extractvalue { ptr, i32 } %17, 1
  store i32 %19, ptr %5, align 4
  br label %20

20:                                               ; preds = %16
  %21 = load i32, ptr %5, align 4
  %22 = call i32 @llvm.eh.typeid.for(ptr @_ZTIb) #2
  %23 = icmp eq i32 %21, %22
  br i1 %23, label %24, label %33

24:                                               ; preds = %20
  %25 = load ptr, ptr %4, align 8
  %26 = call ptr @__cxa_begin_catch(ptr %25) #2
  %27 = load i8, ptr %26, align 1
  %28 = trunc i8 %27 to i1
  %29 = zext i1 %28 to i8
  store i8 %29, ptr %6, align 1
  call void @__cxa_end_catch() #2
  br label %30

30:                                               ; preds = %24
  store i32 0, ptr %2, align 4
  br label %31

31:                                               ; preds = %30, %10
  %32 = load i32, ptr %2, align 4
  ret i32 %32

33:                                               ; preds = %20
  %34 = load ptr, ptr %4, align 8
  %35 = load i32, ptr %5, align 4
  %36 = insertvalue { ptr, i32 } undef, ptr %34, 0
  %37 = insertvalue { ptr, i32 } %36, i32 %35, 1
  resume { ptr, i32 } %37

38:                                               ; preds = %11
  unreachable
}
declare ptr @__cxa_allocate_exception(i64)
declare void @__cxa_throw(ptr, ptr, ptr)
declare i32 @__gxx_personality_v0(...)
; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(ptr) #1
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()

;LINUX-SECTIONS: .section	.text._Z3foob,"ax",@progbits
;LINUX-SECTIONS: _Z3foob:
;LINUX-SECTIONS: .section       .text._Z3foob._Z3foob.__part.{{[0-9]+}},"ax",@progbits
;LINUX-SECTIONS-LABEL: _Z3foob.__part.{{[0-9]+}}:
;LINUX-SECTIONS:        calll   __cxa_begin_catch
