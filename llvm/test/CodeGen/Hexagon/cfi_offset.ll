; RUN: llc -mtriple=hexagon -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck %s
; CHECK: .cfi_def_cfa r30
; CHECK: .cfi_offset r31
; CHECK: .cfi_offset r30

@g0 = global i32 0, align 4
@g1 = external constant ptr

define i32 @f0() personality ptr @f3 {
b0:
  %v0 = alloca i32, align 4
  %v1 = alloca ptr
  %v2 = alloca i32
  %v3 = alloca i32, align 4
  store i32 0, ptr %v0
  %v4 = call ptr @f1(i32 4) #1
  store i32 20, ptr %v4
  invoke void @f2(ptr %v4, ptr @g1, ptr null) #2
          to label %b6 unwind label %b1

b1:                                               ; preds = %b0
  %v6 = landingpad { ptr, i32 }
          catch ptr @g1
  %v7 = extractvalue { ptr, i32 } %v6, 0
  store ptr %v7, ptr %v1
  %v8 = extractvalue { ptr, i32 } %v6, 1
  store i32 %v8, ptr %v2
  br label %b2

b2:                                               ; preds = %b1
  %v9 = load i32, ptr %v2
  %v10 = call i32 @llvm.eh.typeid.for(ptr @g1) #1
  %v11 = icmp eq i32 %v9, %v10
  br i1 %v11, label %b3, label %b5

b3:                                               ; preds = %b2
  %v12 = load ptr, ptr %v1
  %v13 = call ptr @f4(ptr %v12) #1
  %v15 = load i32, ptr %v13, align 4
  store i32 %v15, ptr %v3, align 4
  %v16 = load i32, ptr %v3, align 4
  store i32 %v16, ptr @g0, align 4
  call void @f5() #1
  br label %b4

b4:                                               ; preds = %b3
  %v17 = load i32, ptr @g0, align 4
  ret i32 %v17

b5:                                               ; preds = %b2
  %v18 = load ptr, ptr %v1
  %v19 = load i32, ptr %v2
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
declare i32 @llvm.eh.typeid.for(ptr) #0

declare ptr @f4(ptr)

declare void @f5()

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
attributes #2 = { noreturn }
