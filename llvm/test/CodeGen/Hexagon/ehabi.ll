; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK: GCC_except_table0:
; CHECK: Call site Encoding = uleb128

target triple = "hexagon"

@g0 = external constant i8*

define i32 @f0() #0 personality i8* bitcast (i32 (...)* @f3 to i8*) {
b0:
  %v0 = alloca i32, align 4
  %v1 = alloca i32, align 4
  %v2 = alloca i8*
  %v3 = alloca i32
  %v4 = alloca i32, align 4
  store i32 0, i32* %v0
  store i32 1, i32* %v1, align 4
  %v5 = call i8* @f1(i32 4) #2
  %v6 = bitcast i8* %v5 to i32*
  store i32 20, i32* %v6
  invoke void @f2(i8* %v5, i8* bitcast (i8** @g0 to i8*), i8* null) #3
          to label %b6 unwind label %b1

b1:                                               ; preds = %b0
  %v7 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @g0 to i8*)
  %v8 = extractvalue { i8*, i32 } %v7, 0
  store i8* %v8, i8** %v2
  %v9 = extractvalue { i8*, i32 } %v7, 1
  store i32 %v9, i32* %v3
  br label %b2

b2:                                               ; preds = %b1
  %v10 = load i32, i32* %v3
  %v11 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @g0 to i8*)) #2
  %v12 = icmp eq i32 %v10, %v11
  br i1 %v12, label %b3, label %b5

b3:                                               ; preds = %b2
  %v13 = load i8*, i8** %v2
  %v14 = call i8* @f4(i8* %v13) #2
  %v15 = bitcast i8* %v14 to i32*
  %v16 = load i32, i32* %v15, align 4
  store i32 %v16, i32* %v4, align 4
  store i32 2, i32* %v1, align 4
  call void @f5() #2
  br label %b4

b4:                                               ; preds = %b3
  %v17 = load i32, i32* %v1, align 4
  ret i32 %v17

b5:                                               ; preds = %b2
  %v18 = load i8*, i8** %v2
  %v19 = load i32, i32* %v3
  %v20 = insertvalue { i8*, i32 } undef, i8* %v18, 0
  %v21 = insertvalue { i8*, i32 } %v20, i32 %v19, 1
  resume { i8*, i32 } %v21

b6:                                               ; preds = %b0
  unreachable
}

declare i8* @f1(i32)

declare void @f2(i8*, i8*, i8*)

declare i32 @f3(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #1

declare i8* @f4(i8*)

declare void @f5()

attributes #0 = { "frame-pointer"="all" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
attributes #3 = { noreturn }
