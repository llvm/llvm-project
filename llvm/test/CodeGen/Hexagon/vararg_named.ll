; RUN: llc -mtriple=hexagon -mcpu=hexagonv62 -mtriple=hexagon-unknown-linux-musl -O0 < %s | FileCheck %s

; CHECK-LABEL: foo:

; Check Function prologue.
; Note. All register numbers and offset are fixed.
; Hence, no need of regular expression.

; CHECK: r29 = add(r29,#-16)
; CHECK: r7:6 = memd(r29+#16)
; CHECK: memd(r29+#0) = r7:6
; CHECK: r7:6 = memd(r29+#24)
; CHECK: memd(r29+#8) = r7:6
; CHECK: r7:6 = memd(r29+#32)
; CHECK: memd(r29+#16) = r7:6
; CHECK: r7:6 = memd(r29+#40)
; CHECK: memd(r29+#24) = r7:6
; CHECK: memw(r29+#36) = r3
; CHECK: memw(r29+#40) = r4
; CHECK: memw(r29+#44) = r5
; CHECK: r29 = add(r29,#16)

%struct.AAA = type { i32, i32, i32, i32 }
%struct.__va_list_tag = type { ptr, ptr, ptr }

@aaa = global %struct.AAA { i32 100, i32 200, i32 300, i32 400 }, align 4
@xxx = global %struct.AAA { i32 100, i32 200, i32 300, i32 400 }, align 4
@yyy = global %struct.AAA { i32 100, i32 200, i32 300, i32 400 }, align 4
@ccc = global %struct.AAA { i32 10, i32 20, i32 30, i32 40 }, align 4
@fff = global %struct.AAA { i32 1, i32 2, i32 3, i32 4 }, align 4
@.str = private unnamed_addr constant [13 x i8] c"result = %d\0A\00", align 1

; Function Attrs: nounwind
define i32 @foo(i32 %xx, i32 %z, i32 %m, ptr byval(%struct.AAA) align 4 %bbb, ptr byval(%struct.AAA) align 4 %GGG, ...) #0 {
entry:
  %xx.addr = alloca i32, align 4
  %z.addr = alloca i32, align 4
  %m.addr = alloca i32, align 4
  %ap = alloca [1 x %struct.__va_list_tag], align 8
  %d = alloca i32, align 4
  %ret = alloca i32, align 4
  %ddd = alloca %struct.AAA, align 4
  %ggg = alloca %struct.AAA, align 4
  %nnn = alloca %struct.AAA, align 4
  store i32 %xx, ptr %xx.addr, align 4
  store i32 %z, ptr %z.addr, align 4
  store i32 %m, ptr %m.addr, align 4
  store i32 0, ptr %ret, align 4
  call void @llvm.va_start(ptr %ap)
  %d2 = getelementptr inbounds %struct.AAA, ptr %bbb, i32 0, i32 3
  %0 = load i32, ptr %d2, align 4
  %1 = load i32, ptr %ret, align 4
  %add = add nsw i32 %1, %0
  store i32 %add, ptr %ret, align 4
  %2 = load i32, ptr %z.addr, align 4
  %3 = load i32, ptr %ret, align 4
  %add3 = add nsw i32 %3, %2
  store i32 %add3, ptr %ret, align 4
  br label %vaarg.maybe_reg

vaarg.maybe_reg:                                  ; preds = %entry
  %__current_saved_reg_area_pointer = load ptr, ptr %ap
  %__saved_reg_area_end_pointer_p = getelementptr inbounds %struct.__va_list_tag, ptr %ap, i32 0, i32 1
  %__saved_reg_area_end_pointer = load ptr, ptr %__saved_reg_area_end_pointer_p
  %__new_saved_reg_area_pointer = getelementptr i8, ptr %__current_saved_reg_area_pointer, i32 4
  %4 = icmp sgt ptr %__new_saved_reg_area_pointer, %__saved_reg_area_end_pointer
  br i1 %4, label %vaarg.on_stack, label %vaarg.in_reg

vaarg.in_reg:                                     ; preds = %vaarg.maybe_reg
  store ptr %__new_saved_reg_area_pointer, ptr %ap
  br label %vaarg.end

vaarg.on_stack:                                   ; preds = %vaarg.maybe_reg
  %__overflow_area_pointer_p = getelementptr inbounds %struct.__va_list_tag, ptr %ap, i32 0, i32 2
  %__overflow_area_pointer = load ptr, ptr %__overflow_area_pointer_p
  %__overflow_area_pointer.next = getelementptr i8, ptr %__overflow_area_pointer, i32 4
  store ptr %__overflow_area_pointer.next, ptr %__overflow_area_pointer_p
  store ptr %__overflow_area_pointer.next, ptr %ap
  br label %vaarg.end

vaarg.end:                                        ; preds = %vaarg.on_stack, %vaarg.in_reg
  %vaarg.addr = phi ptr [ %__current_saved_reg_area_pointer, %vaarg.in_reg ], [ %__overflow_area_pointer, %vaarg.on_stack ]
  %5 = load i32, ptr %vaarg.addr
  store i32 %5, ptr %d, align 4
  %6 = load i32, ptr %d, align 4
  %7 = load i32, ptr %ret, align 4
  %add5 = add nsw i32 %7, %6
  store i32 %add5, ptr %ret, align 4
  %__overflow_area_pointer_p7 = getelementptr inbounds %struct.__va_list_tag, ptr %ap, i32 0, i32 2
  %__overflow_area_pointer8 = load ptr, ptr %__overflow_area_pointer_p7
  %__overflow_area_pointer.next9 = getelementptr i8, ptr %__overflow_area_pointer8, i32 16
  store ptr %__overflow_area_pointer.next9, ptr %__overflow_area_pointer_p7
  call void @llvm.memcpy.p0.p0.i32(ptr %ddd, ptr %__overflow_area_pointer8, i32 16, i32 4, i1 false)
  %d10 = getelementptr inbounds %struct.AAA, ptr %ddd, i32 0, i32 3
  %8 = load i32, ptr %d10, align 4
  %9 = load i32, ptr %ret, align 4
  %add11 = add nsw i32 %9, %8
  store i32 %add11, ptr %ret, align 4
  %__overflow_area_pointer_p13 = getelementptr inbounds %struct.__va_list_tag, ptr %ap, i32 0, i32 2
  %__overflow_area_pointer14 = load ptr, ptr %__overflow_area_pointer_p13
  %__overflow_area_pointer.next15 = getelementptr i8, ptr %__overflow_area_pointer14, i32 16
  store ptr %__overflow_area_pointer.next15, ptr %__overflow_area_pointer_p13
  call void @llvm.memcpy.p0.p0.i32(ptr %ggg, ptr %__overflow_area_pointer14, i32 16, i32 4, i1 false)
  %d16 = getelementptr inbounds %struct.AAA, ptr %ggg, i32 0, i32 3
  %10 = load i32, ptr %d16, align 4
  %11 = load i32, ptr %ret, align 4
  %add17 = add nsw i32 %11, %10
  store i32 %add17, ptr %ret, align 4
  %__overflow_area_pointer_p19 = getelementptr inbounds %struct.__va_list_tag, ptr %ap, i32 0, i32 2
  %__overflow_area_pointer20 = load ptr, ptr %__overflow_area_pointer_p19
  %__overflow_area_pointer.next21 = getelementptr i8, ptr %__overflow_area_pointer20, i32 16
  store ptr %__overflow_area_pointer.next21, ptr %__overflow_area_pointer_p19
  call void @llvm.memcpy.p0.p0.i32(ptr %nnn, ptr %__overflow_area_pointer20, i32 16, i32 4, i1 false)
  %d22 = getelementptr inbounds %struct.AAA, ptr %nnn, i32 0, i32 3
  %12 = load i32, ptr %d22, align 4
  %13 = load i32, ptr %ret, align 4
  %add23 = add nsw i32 %13, %12
  store i32 %add23, ptr %ret, align 4
  br label %vaarg.maybe_reg25

vaarg.maybe_reg25:                                ; preds = %vaarg.end
  %__current_saved_reg_area_pointer27 = load ptr, ptr %ap
  %__saved_reg_area_end_pointer_p28 = getelementptr inbounds %struct.__va_list_tag, ptr %ap, i32 0, i32 1
  %__saved_reg_area_end_pointer29 = load ptr, ptr %__saved_reg_area_end_pointer_p28
  %__new_saved_reg_area_pointer30 = getelementptr i8, ptr %__current_saved_reg_area_pointer27, i32 4
  %14 = icmp sgt ptr %__new_saved_reg_area_pointer30, %__saved_reg_area_end_pointer29
  br i1 %14, label %vaarg.on_stack32, label %vaarg.in_reg31

vaarg.in_reg31:                                   ; preds = %vaarg.maybe_reg25
  store ptr %__new_saved_reg_area_pointer30, ptr %ap
  br label %vaarg.end36

vaarg.on_stack32:                                 ; preds = %vaarg.maybe_reg25
  %__overflow_area_pointer_p33 = getelementptr inbounds %struct.__va_list_tag, ptr %ap, i32 0, i32 2
  %__overflow_area_pointer34 = load ptr, ptr %__overflow_area_pointer_p33
  %__overflow_area_pointer.next35 = getelementptr i8, ptr %__overflow_area_pointer34, i32 4
  store ptr %__overflow_area_pointer.next35, ptr %__overflow_area_pointer_p33
  store ptr %__overflow_area_pointer.next35, ptr %ap
  br label %vaarg.end36

vaarg.end36:                                      ; preds = %vaarg.on_stack32, %vaarg.in_reg31
  %vaarg.addr37 = phi ptr [ %__current_saved_reg_area_pointer27, %vaarg.in_reg31 ], [ %__overflow_area_pointer34, %vaarg.on_stack32 ]
  %15 = load i32, ptr %vaarg.addr37
  store i32 %15, ptr %d, align 4
  %16 = load i32, ptr %d, align 4
  %17 = load i32, ptr %ret, align 4
  %add38 = add nsw i32 %17, %16
  store i32 %add38, ptr %ret, align 4
  %18 = load i32, ptr %m.addr, align 4
  %19 = load i32, ptr %ret, align 4
  %add39 = add nsw i32 %19, %18
  store i32 %add39, ptr %ret, align 4
  call void @llvm.va_end(ptr %ap)
  %20 = load i32, ptr %ret, align 4
  ret i32 %20
}

; Function Attrs: nounwind
declare void @llvm.va_start(ptr) #1

; Function Attrs: nounwind
declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture readonly, i32, i32, i1) #1

; Function Attrs: nounwind
declare void @llvm.va_end(ptr) #1

; Function Attrs: nounwind
define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %x = alloca i32, align 4
  store i32 0, ptr %retval
  %call = call i32 (i32, i32, i32, ptr, ptr, ...) @foo(i32 1, i32 3, i32 5, ptr byval(%struct.AAA) align 4 @aaa, ptr byval(%struct.AAA) align 4 @fff, i32 2, ptr byval(%struct.AAA) align 4 @xxx, ptr byval(%struct.AAA) align 4 @yyy, ptr byval(%struct.AAA) align 4 @ccc, i32 4)
  store i32 %call, ptr %x, align 4
  %0 = load i32, ptr %x, align 4
  %call1 = call i32 (ptr, ...) @printf(ptr @.str, i32 %0)
  %1 = load i32, ptr %x, align 4
  ret i32 %1
}

declare i32 @printf(ptr, ...) #2

attributes #0 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"Clang 3.1"}
