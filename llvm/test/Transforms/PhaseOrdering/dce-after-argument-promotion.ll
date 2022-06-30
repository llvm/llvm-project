; RUN: opt -O3 -S < %s | FileCheck %s

; Arg promotion eliminates the struct argument but may leave dead arguments after its work

%struct.ss = type { i32, i64 }

@dummy = global i32 0
; CHECK: [[DUMMY:@.*]] = local_unnamed_addr global i32 0

define internal void @f(%struct.ss* byval(%struct.ss) align 8 %b, i32* byval(i32) align 4 %X) noinline nounwind  {
; CHECK-LABEL: define {{[^@]+}}@f
; CHECK-SAME: (i32 [[B_0:%.*]], i32 [[X:%.*]]){{[^#]*}} #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TEMP:%.*]] = add i32 [[B_0]], 1
; CHECK-NEXT:    store i32 [[TEMP]], i32* [[DUMMY]], align 4
; CHECK-NEXT:    ret void
;
entry:
  %temp = getelementptr %struct.ss, %struct.ss* %b, i32 0, i32 0
  %temp1 = load i32, i32* %temp, align 4
  %temp2 = add i32 %temp1, 1
  store i32 %temp2, i32* @dummy
  store i32 %temp2, i32* %X
  ret void
}

define i32 @test(i32* %X) {
; CHECK-LABEL: define {{[^@]+}}@test
; CHECK-SAME: (i32* {{[^%]*}} [[X:%.*]]){{[^#]*}} #[[ATTR1:[0-9]+]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[X_VAL:%.*]] = load i32, i32* [[X]], align 4
; CHECK-NEXT:    tail call {{.*}}void @f(i32 1, i32 [[X_VAL]])
; CHECK-NEXT:    ret i32 0
;
entry:
  %S = alloca %struct.ss, align 8
  %temp1 = getelementptr %struct.ss, %struct.ss* %S, i32 0, i32 0
  store i32 1, i32* %temp1, align 8
  %temp4 = getelementptr %struct.ss, %struct.ss* %S, i32 0, i32 1
  store i64 2, i64* %temp4, align 4
  call void @f( %struct.ss* byval(%struct.ss) align 8 %S, i32* byval(i32) align 4 %X)
  ret i32 0
}
