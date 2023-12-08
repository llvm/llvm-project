; RUN: opt -O3 -S < %s | FileCheck %s

; Arg promotion eliminates the struct argument but may leave dead arguments after its work

%struct.ss = type { i32, i64 }

@dummy = global i32 0
; CHECK: [[DUMMY:@.*]] = local_unnamed_addr global i32 0

define internal void @f(ptr byval(%struct.ss) align 8 %b, ptr byval(i32) align 4 %X) noinline nounwind  {
; CHECK-LABEL: define {{[^@]+}}@f
; CHECK-SAME: (i32 [[B_0:%.*]]){{[^#]*}} #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TEMP:%.*]] = add i32 [[B_0]], 1
; CHECK-NEXT:    store i32 [[TEMP]], ptr [[DUMMY]], align 4
; CHECK-NEXT:    ret void
;
entry:
  %temp1 = load i32, ptr %b, align 4
  %temp2 = add i32 %temp1, 1
  store i32 %temp2, ptr @dummy
  store i32 %temp2, ptr %X
  ret void
}

define i32 @test(ptr %X) {
; CHECK-LABEL: define {{[^@]+}}@test
; CHECK-SAME: (ptr {{[^%]*}} [[X:%.*]]){{[^#]*}} #[[ATTR1:[0-9]+]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    tail call {{.*}}void @f(i32 1)
; CHECK-NEXT:    ret i32 0
;
entry:
  %S = alloca %struct.ss, align 8
  store i32 1, ptr %S, align 8
  %temp4 = getelementptr %struct.ss, ptr %S, i32 0, i32 1
  store i64 2, ptr %temp4, align 4
  call void @f( ptr byval(%struct.ss) align 8 %S, ptr byval(i32) align 4 %X)
  ret i32 0
}
