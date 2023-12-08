; RUN: llc -march=hexagon -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck %s

; CHECK: GCC_except_table0:
; CHECK: Call site Encoding = uleb128

target triple = "hexagon"

@g0 = external constant ptr

define i32 @f0() #0 personality ptr @f3 {
b0:
  %v0 = alloca i32, align 4
  %v1 = alloca i32, align 4
  %v2 = alloca ptr
  %v3 = alloca i32
  %v4 = alloca i32, align 4
  store i32 0, ptr %v0
  store i32 1, ptr %v1, align 4
  %v5 = call ptr @f1(i32 4) #2
  store i32 20, ptr %v5
  invoke void @f2(ptr %v5, ptr @g0, ptr null) #3
          to label %b6 unwind label %b1

b1:                                               ; preds = %b0
  %v7 = landingpad { ptr, i32 }
          catch ptr @g0
  %v8 = extractvalue { ptr, i32 } %v7, 0
  store ptr %v8, ptr %v2
  %v9 = extractvalue { ptr, i32 } %v7, 1
  store i32 %v9, ptr %v3
  br label %b2

b2:                                               ; preds = %b1
  %v10 = load i32, ptr %v3
  %v11 = call i32 @llvm.eh.typeid.for(ptr @g0) #2
  %v12 = icmp eq i32 %v10, %v11
  br i1 %v12, label %b3, label %b5

b3:                                               ; preds = %b2
  %v13 = load ptr, ptr %v2
  %v14 = call ptr @f4(ptr %v13) #2
  %v16 = load i32, ptr %v14, align 4
  store i32 %v16, ptr %v4, align 4
  store i32 2, ptr %v1, align 4
  call void @f5() #2
  br label %b4

b4:                                               ; preds = %b3
  %v17 = load i32, ptr %v1, align 4
  ret i32 %v17

b5:                                               ; preds = %b2
  %v18 = load ptr, ptr %v2
  %v19 = load i32, ptr %v3
  %v20 = insertvalue { ptr, i32 } undef, ptr %v18, 0
  %v21 = insertvalue { ptr, i32 } %v20, i32 %v19, 1
  resume { ptr, i32 } %v21

b6:                                               ; preds = %b0
  unreachable
}

declare ptr @f1(i32)

declare void @f2(ptr, ptr, ptr)

declare i32 @f3(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(ptr) #1

declare ptr @f4(ptr)

declare void @f5()

attributes #0 = { "frame-pointer"="all" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
attributes #3 = { noreturn }
