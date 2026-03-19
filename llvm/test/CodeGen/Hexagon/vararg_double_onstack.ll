; RUN: llc -mtriple=hexagon -mcpu=hexagonv62 -mtriple=hexagon-unknown-linux-musl -O0 < %s | FileCheck %s

; CHECK-LABEL: foo:

; Check Function prologue.
; Note. All register numbers and offset are fixed.
; Hence, no need of regular expression.

; CHECK: r29 = add(r29,#-8)
; CHECK: memw(r29+#4) = r5
; CHECK: r29 = add(r29,#8)

%struct.AAA = type { i32, i32, i32, i32 }
%struct.__va_list_tag = type { ptr, ptr, ptr }

@aaa = global %struct.AAA { i32 100, i32 200, i32 300, i32 400 }, align 4
@.str = private unnamed_addr constant [13 x i8] c"result = %d\0A\00", align 1

; Function Attrs: nounwind
define i32 @foo(i32 %xx, i32 %a, i32 %b, i32 %c, i32 %x, ...) #0 {
entry:
  %xx.addr = alloca i32, align 4
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %c.addr = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %ap = alloca [1 x %struct.__va_list_tag], align 8
  %d = alloca i32, align 4
  %ret = alloca i32, align 4
  %bbb = alloca %struct.AAA, align 4
  store i32 %xx, ptr %xx.addr, align 4
  store i32 %a, ptr %a.addr, align 4
  store i32 %b, ptr %b.addr, align 4
  store i32 %c, ptr %c.addr, align 4
  store i32 %x, ptr %x.addr, align 4
  store i32 0, ptr %ret, align 4
  call void @llvm.va_start(ptr %ap)
  br label %vaarg.maybe_reg

vaarg.maybe_reg:                                  ; preds = %entry
  %__current_saved_reg_area_pointer = load ptr, ptr %ap
  %__saved_reg_area_end_pointer_p = getelementptr inbounds %struct.__va_list_tag, ptr %ap, i32 0, i32 1
  %__saved_reg_area_end_pointer = load ptr, ptr %__saved_reg_area_end_pointer_p
  %0 = ptrtoint ptr %__current_saved_reg_area_pointer to i32
  %align_current_saved_reg_area_pointer = add i32 %0, 7
  %align_current_saved_reg_area_pointer3 = and i32 %align_current_saved_reg_area_pointer, -8
  %align_current_saved_reg_area_pointer4 = inttoptr i32 %align_current_saved_reg_area_pointer3 to ptr
  %__new_saved_reg_area_pointer = getelementptr i8, ptr %align_current_saved_reg_area_pointer4, i32 8
  %1 = icmp sgt ptr %__new_saved_reg_area_pointer, %__saved_reg_area_end_pointer
  br i1 %1, label %vaarg.on_stack, label %vaarg.in_reg

vaarg.in_reg:                                     ; preds = %vaarg.maybe_reg
  store ptr %__new_saved_reg_area_pointer, ptr %ap
  br label %vaarg.end

vaarg.on_stack:                                   ; preds = %vaarg.maybe_reg
  %__overflow_area_pointer_p = getelementptr inbounds %struct.__va_list_tag, ptr %ap, i32 0, i32 2
  %__overflow_area_pointer = load ptr, ptr %__overflow_area_pointer_p
  %2 = ptrtoint ptr %__overflow_area_pointer to i32
  %align_overflow_area_pointer = add i32 %2, 7
  %align_overflow_area_pointer5 = and i32 %align_overflow_area_pointer, -8
  %align_overflow_area_pointer6 = inttoptr i32 %align_overflow_area_pointer5 to ptr
  %__overflow_area_pointer.next = getelementptr i8, ptr %align_overflow_area_pointer6, i32 8
  store ptr %__overflow_area_pointer.next, ptr %__overflow_area_pointer_p
  store ptr %__overflow_area_pointer.next, ptr %ap
  br label %vaarg.end

vaarg.end:                                        ; preds = %vaarg.on_stack, %vaarg.in_reg
  %vaarg.addr = phi ptr [ %align_current_saved_reg_area_pointer4, %vaarg.in_reg ], [ %align_overflow_area_pointer6, %vaarg.on_stack ]
  %3 = load i64, ptr %vaarg.addr
  %conv = trunc i64 %3 to i32
  store i32 %conv, ptr %d, align 4
  %4 = load i32, ptr %d, align 4
  %5 = load i32, ptr %ret, align 4
  %add = add nsw i32 %5, %4
  store i32 %add, ptr %ret, align 4
  %__overflow_area_pointer_p8 = getelementptr inbounds %struct.__va_list_tag, ptr %ap, i32 0, i32 2
  %__overflow_area_pointer9 = load ptr, ptr %__overflow_area_pointer_p8
  %__overflow_area_pointer.next10 = getelementptr i8, ptr %__overflow_area_pointer9, i32 16
  store ptr %__overflow_area_pointer.next10, ptr %__overflow_area_pointer_p8
  call void @llvm.memcpy.p0.p0.i32(ptr %bbb, ptr %__overflow_area_pointer9, i32 16, i32 4, i1 false)
  %d11 = getelementptr inbounds %struct.AAA, ptr %bbb, i32 0, i32 3
  %6 = load i32, ptr %d11, align 4
  %7 = load i32, ptr %ret, align 4
  %add12 = add nsw i32 %7, %6
  store i32 %add12, ptr %ret, align 4
  br label %vaarg.maybe_reg14

vaarg.maybe_reg14:                                ; preds = %vaarg.end
  %__current_saved_reg_area_pointer16 = load ptr, ptr %ap
  %__saved_reg_area_end_pointer_p17 = getelementptr inbounds %struct.__va_list_tag, ptr %ap, i32 0, i32 1
  %__saved_reg_area_end_pointer18 = load ptr, ptr %__saved_reg_area_end_pointer_p17
  %__new_saved_reg_area_pointer19 = getelementptr i8, ptr %__current_saved_reg_area_pointer16, i32 4
  %8 = icmp sgt ptr %__new_saved_reg_area_pointer19, %__saved_reg_area_end_pointer18
  br i1 %8, label %vaarg.on_stack21, label %vaarg.in_reg20

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
  %9 = load i32, ptr %vaarg.addr26
  store i32 %9, ptr %d, align 4
  %10 = load i32, ptr %d, align 4
  %11 = load i32, ptr %ret, align 4
  %add27 = add nsw i32 %11, %10
  store i32 %add27, ptr %ret, align 4
  br label %vaarg.maybe_reg29

vaarg.maybe_reg29:                                ; preds = %vaarg.end25
  %__current_saved_reg_area_pointer31 = load ptr, ptr %ap
  %__saved_reg_area_end_pointer_p32 = getelementptr inbounds %struct.__va_list_tag, ptr %ap, i32 0, i32 1
  %__saved_reg_area_end_pointer33 = load ptr, ptr %__saved_reg_area_end_pointer_p32
  %12 = ptrtoint ptr %__current_saved_reg_area_pointer31 to i32
  %align_current_saved_reg_area_pointer34 = add i32 %12, 7
  %align_current_saved_reg_area_pointer35 = and i32 %align_current_saved_reg_area_pointer34, -8
  %align_current_saved_reg_area_pointer36 = inttoptr i32 %align_current_saved_reg_area_pointer35 to ptr
  %__new_saved_reg_area_pointer37 = getelementptr i8, ptr %align_current_saved_reg_area_pointer36, i32 8
  %13 = icmp sgt ptr %__new_saved_reg_area_pointer37, %__saved_reg_area_end_pointer33
  br i1 %13, label %vaarg.on_stack39, label %vaarg.in_reg38

vaarg.in_reg38:                                   ; preds = %vaarg.maybe_reg29
  store ptr %__new_saved_reg_area_pointer37, ptr %ap
  br label %vaarg.end46

vaarg.on_stack39:                                 ; preds = %vaarg.maybe_reg29
  %__overflow_area_pointer_p40 = getelementptr inbounds %struct.__va_list_tag, ptr %ap, i32 0, i32 2
  %__overflow_area_pointer41 = load ptr, ptr %__overflow_area_pointer_p40
  %14 = ptrtoint ptr %__overflow_area_pointer41 to i32
  %align_overflow_area_pointer42 = add i32 %14, 7
  %align_overflow_area_pointer43 = and i32 %align_overflow_area_pointer42, -8
  %align_overflow_area_pointer44 = inttoptr i32 %align_overflow_area_pointer43 to ptr
  %__overflow_area_pointer.next45 = getelementptr i8, ptr %align_overflow_area_pointer44, i32 8
  store ptr %__overflow_area_pointer.next45, ptr %__overflow_area_pointer_p40
  store ptr %__overflow_area_pointer.next45, ptr %ap
  br label %vaarg.end46

vaarg.end46:                                      ; preds = %vaarg.on_stack39, %vaarg.in_reg38
  %vaarg.addr47 = phi ptr [ %align_current_saved_reg_area_pointer36, %vaarg.in_reg38 ], [ %align_overflow_area_pointer44, %vaarg.on_stack39 ]
  %15 = load i64, ptr %vaarg.addr47
  %conv48 = trunc i64 %15 to i32
  store i32 %conv48, ptr %d, align 4
  %16 = load i32, ptr %d, align 4
  %17 = load i32, ptr %ret, align 4
  %add49 = add nsw i32 %17, %16
  store i32 %add49, ptr %ret, align 4
  call void @llvm.va_end(ptr %ap)
  %18 = load i32, ptr %ret, align 4
  ret i32 %18
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
  %y = alloca i64, align 8
  store i32 0, ptr %retval
  store i64 1000000, ptr %y, align 8
  %0 = load i64, ptr %y, align 8
  %1 = load i64, ptr %y, align 8
  %call = call i32 (i32, i32, i32, i32, i32, ...) @foo(i32 1, i32 2, i32 3, i32 4, i32 5, i64 %0, ptr byval(%struct.AAA) align 4 @aaa, i32 4, i64 %1)
  store i32 %call, ptr %x, align 4
  %2 = load i32, ptr %x, align 4
  %call1 = call i32 (ptr, ...) @printf(ptr @.str, i32 %2)
  %3 = load i32, ptr %x, align 4
  ret i32 %3
}

declare i32 @printf(ptr, ...) #2

attributes #0 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"Clang 3.1"}
