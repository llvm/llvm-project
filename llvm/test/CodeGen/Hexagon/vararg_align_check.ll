; RUN: llc -mtriple=hexagon -mcpu=hexagonv62 -mtriple=hexagon-unknown-linux-musl -O0 < %s | FileCheck %s

; CHECK-LABEL: foo:

; Check Function prologue.
; Note. All register numbers and offset are fixed.
; Hence, no need of regular expression.

; CHECK: r29 = add(r29,#-24)
; CHECK: r7:6 = memd(r29+#24)
; CHECK: memd(r29+#0) = r7:6
; CHECK: r7:6 = memd(r29+#32)
; CHECK: memd(r29+#8) = r7:6
; CHECK: r7:6 = memd(r29+#40)
; CHECK: memd(r29+#16) = r7:6
; CHECK: memw(r29+#28) = r1
; CHECK: memw(r29+#32) = r2
; CHECK: memw(r29+#36) = r3
; CHECK: memw(r29+#40) = r4
; CHECK: memw(r29+#44) = r5
; CHECK: r29 = add(r29,#24)

%struct.AAA = type { i32, i32, i32, i32 }
%struct.BBB = type { i8, i64, i32 }
%struct.__va_list_tag = type { ptr, ptr, ptr }

@aaa = global %struct.AAA { i32 100, i32 200, i32 300, i32 400 }, align 4
@ddd = global { i8, i64, i32, [4 x i8] } { i8 1, i64 1000000, i32 5, [4 x i8] undef }, align 8
@.str = private unnamed_addr constant [13 x i8] c"result = %d\0A\00", align 1

; Function Attrs: nounwind
define i32 @foo(i32 %xx, ptr byval(%struct.BBB) align 8 %eee, ...) #0 {
entry:
  %xx.addr = alloca i32, align 4
  %ap = alloca [1 x %struct.__va_list_tag], align 8
  %d = alloca i32, align 4
  %k = alloca i64, align 8
  %ret = alloca i32, align 4
  %bbb = alloca %struct.AAA, align 4
  store i32 %xx, ptr %xx.addr, align 4
  store i32 0, ptr %ret, align 4
  %0 = load i8, ptr %eee, align 1
  %tobool = trunc i8 %0 to i1
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 1, ptr %ret, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call void @llvm.va_start(ptr %ap)
  br label %vaarg.maybe_reg

vaarg.maybe_reg:                                  ; preds = %if.end
  %__current_saved_reg_area_pointer = load ptr, ptr %ap
  %__saved_reg_area_end_pointer_p = getelementptr inbounds %struct.__va_list_tag, ptr %ap, i32 0, i32 1
  %__saved_reg_area_end_pointer = load ptr, ptr %__saved_reg_area_end_pointer_p
  %1 = ptrtoint ptr %__current_saved_reg_area_pointer to i32
  %align_current_saved_reg_area_pointer = add i32 %1, 7
  %align_current_saved_reg_area_pointer3 = and i32 %align_current_saved_reg_area_pointer, -8
  %align_current_saved_reg_area_pointer4 = inttoptr i32 %align_current_saved_reg_area_pointer3 to ptr
  %__new_saved_reg_area_pointer = getelementptr i8, ptr %align_current_saved_reg_area_pointer4, i32 8
  %2 = icmp sgt ptr %__new_saved_reg_area_pointer, %__saved_reg_area_end_pointer
  br i1 %2, label %vaarg.on_stack, label %vaarg.in_reg

vaarg.in_reg:                                     ; preds = %vaarg.maybe_reg
  store ptr %__new_saved_reg_area_pointer, ptr %ap
  br label %vaarg.end

vaarg.on_stack:                                   ; preds = %vaarg.maybe_reg
  %__overflow_area_pointer_p = getelementptr inbounds %struct.__va_list_tag, ptr %ap, i32 0, i32 2
  %__overflow_area_pointer = load ptr, ptr %__overflow_area_pointer_p
  %3 = ptrtoint ptr %__overflow_area_pointer to i32
  %align_overflow_area_pointer = add i32 %3, 7
  %align_overflow_area_pointer5 = and i32 %align_overflow_area_pointer, -8
  %align_overflow_area_pointer6 = inttoptr i32 %align_overflow_area_pointer5 to ptr
  %__overflow_area_pointer.next = getelementptr i8, ptr %align_overflow_area_pointer6, i32 8
  store ptr %__overflow_area_pointer.next, ptr %__overflow_area_pointer_p
  store ptr %__overflow_area_pointer.next, ptr %ap
  br label %vaarg.end

vaarg.end:                                        ; preds = %vaarg.on_stack, %vaarg.in_reg
  %vaarg.addr = phi ptr [ %align_current_saved_reg_area_pointer4, %vaarg.in_reg ], [ %align_overflow_area_pointer6, %vaarg.on_stack ]
  %4 = load i64, ptr %vaarg.addr
  store i64 %4, ptr %k, align 8
  %5 = load i64, ptr %k, align 8
  %conv = trunc i64 %5 to i32
  %div = sdiv i32 %conv, 1000
  %6 = load i32, ptr %ret, align 4
  %add = add nsw i32 %6, %div
  store i32 %add, ptr %ret, align 4
  %__overflow_area_pointer_p8 = getelementptr inbounds %struct.__va_list_tag, ptr %ap, i32 0, i32 2
  %__overflow_area_pointer9 = load ptr, ptr %__overflow_area_pointer_p8
  %__overflow_area_pointer.next10 = getelementptr i8, ptr %__overflow_area_pointer9, i32 16
  store ptr %__overflow_area_pointer.next10, ptr %__overflow_area_pointer_p8
  call void @llvm.memcpy.p0.p0.i32(ptr %bbb, ptr %__overflow_area_pointer9, i32 16, i32 4, i1 false)
  %d11 = getelementptr inbounds %struct.AAA, ptr %bbb, i32 0, i32 3
  %7 = load i32, ptr %d11, align 4
  %8 = load i32, ptr %ret, align 4
  %add12 = add nsw i32 %8, %7
  store i32 %add12, ptr %ret, align 4
  br label %vaarg.maybe_reg14

vaarg.maybe_reg14:                                ; preds = %vaarg.end
  %__current_saved_reg_area_pointer16 = load ptr, ptr %ap
  %__saved_reg_area_end_pointer_p17 = getelementptr inbounds %struct.__va_list_tag, ptr %ap, i32 0, i32 1
  %__saved_reg_area_end_pointer18 = load ptr, ptr %__saved_reg_area_end_pointer_p17
  %__new_saved_reg_area_pointer19 = getelementptr i8, ptr %__current_saved_reg_area_pointer16, i32 4
  %9 = icmp sgt ptr %__new_saved_reg_area_pointer19, %__saved_reg_area_end_pointer18
  br i1 %9, label %vaarg.on_stack21, label %vaarg.in_reg20

vaarg.in_reg20:                                   ; preds = %vaarg.maybe_reg14
  store ptr %__new_saved_reg_area_pointer19, ptr %ap
  br label %vaarg.end25

vaarg.on_stack21:                                 ; preds = %vaarg.maybe_reg14
  %__overflow_area_pointer_p22 = getelementptr inbounds %struct.__va_list_tag, ptr %ap, i32 0, i32 2
  %__overflow_area_pointer23 = load ptr, ptr %__overflow_area_pointer_p22
  %__overflow_area_pointer.next24 = getelementptr i8, ptr %__overflow_area_pointer23, i32 4
  store ptr %__overflow_area_pointer.next24, ptr %__overflow_area_pointer_p22
  store ptr %__overflow_area_pointer.next24, ptr %ap
  br label %vaarg.end25

vaarg.end25:                                      ; preds = %vaarg.on_stack21, %vaarg.in_reg20
  %vaarg.addr26 = phi ptr [ %__current_saved_reg_area_pointer16, %vaarg.in_reg20 ], [ %__overflow_area_pointer23, %vaarg.on_stack21 ]
  %10 = load i32, ptr %vaarg.addr26
  store i32 %10, ptr %d, align 4
  %11 = load i32, ptr %d, align 4
  %12 = load i32, ptr %ret, align 4
  %add27 = add nsw i32 %12, %11
  store i32 %add27, ptr %ret, align 4
  call void @llvm.va_end(ptr %ap)
  %13 = load i32, ptr %ret, align 4
  ret i32 %13
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
  %m = alloca i64, align 8
  store i32 0, ptr %retval
  store i64 1000000, ptr %m, align 8
  %0 = load i64, ptr %m, align 8
  %call = call i32 (i32, ptr, ...) @foo(i32 1, ptr byval(%struct.BBB) align 8 @ddd, i64 %0, ptr byval(%struct.AAA) align 4 @aaa, i32 4)
  store i32 %call, ptr %x, align 4
  %1 = load i32, ptr %x, align 4
  %call1 = call i32 (ptr, ...) @printf(ptr @.str, i32 %1)
  %2 = load i32, ptr %x, align 4
  ret i32 %2
}

declare i32 @printf(ptr, ...) #2

attributes #1 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"Clang 3.1"}
