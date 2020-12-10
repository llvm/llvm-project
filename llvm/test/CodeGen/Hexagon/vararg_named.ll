; RUN: llc -march=hexagon -mcpu=hexagonv62 -mtriple=hexagon-unknown-linux-musl -O0 < %s | FileCheck %s

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
%struct.__va_list_tag = type { i8*, i8*, i8* }

@aaa = global %struct.AAA { i32 100, i32 200, i32 300, i32 400 }, align 4
@xxx = global %struct.AAA { i32 100, i32 200, i32 300, i32 400 }, align 4
@yyy = global %struct.AAA { i32 100, i32 200, i32 300, i32 400 }, align 4
@ccc = global %struct.AAA { i32 10, i32 20, i32 30, i32 40 }, align 4
@fff = global %struct.AAA { i32 1, i32 2, i32 3, i32 4 }, align 4
@.str = private unnamed_addr constant [13 x i8] c"result = %d\0A\00", align 1

; Function Attrs: nounwind
define i32 @foo(i32 %xx, i32 %z, i32 %m, %struct.AAA* byval(%struct.AAA) align 4 %bbb, %struct.AAA* byval(%struct.AAA) align 4 %GGG, ...) #0 {
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
  store i32 %xx, i32* %xx.addr, align 4
  store i32 %z, i32* %z.addr, align 4
  store i32 %m, i32* %m.addr, align 4
  store i32 0, i32* %ret, align 4
  %arraydecay = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0
  %arraydecay1 = bitcast %struct.__va_list_tag* %arraydecay to i8*
  call void @llvm.va_start(i8* %arraydecay1)
  %d2 = getelementptr inbounds %struct.AAA, %struct.AAA* %bbb, i32 0, i32 3
  %0 = load i32, i32* %d2, align 4
  %1 = load i32, i32* %ret, align 4
  %add = add nsw i32 %1, %0
  store i32 %add, i32* %ret, align 4
  %2 = load i32, i32* %z.addr, align 4
  %3 = load i32, i32* %ret, align 4
  %add3 = add nsw i32 %3, %2
  store i32 %add3, i32* %ret, align 4
  %arraydecay4 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0
  br label %vaarg.maybe_reg

vaarg.maybe_reg:                                  ; preds = %entry
  %__current_saved_reg_area_pointer_p = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %arraydecay4, i32 0, i32 0
  %__current_saved_reg_area_pointer = load i8*, i8** %__current_saved_reg_area_pointer_p
  %__saved_reg_area_end_pointer_p = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %arraydecay4, i32 0, i32 1
  %__saved_reg_area_end_pointer = load i8*, i8** %__saved_reg_area_end_pointer_p
  %__new_saved_reg_area_pointer = getelementptr i8, i8* %__current_saved_reg_area_pointer, i32 4
  %4 = icmp sgt i8* %__new_saved_reg_area_pointer, %__saved_reg_area_end_pointer
  br i1 %4, label %vaarg.on_stack, label %vaarg.in_reg

vaarg.in_reg:                                     ; preds = %vaarg.maybe_reg
  %5 = bitcast i8* %__current_saved_reg_area_pointer to i32*
  store i8* %__new_saved_reg_area_pointer, i8** %__current_saved_reg_area_pointer_p
  br label %vaarg.end

vaarg.on_stack:                                   ; preds = %vaarg.maybe_reg
  %__overflow_area_pointer_p = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %arraydecay4, i32 0, i32 2
  %__overflow_area_pointer = load i8*, i8** %__overflow_area_pointer_p
  %__overflow_area_pointer.next = getelementptr i8, i8* %__overflow_area_pointer, i32 4
  store i8* %__overflow_area_pointer.next, i8** %__overflow_area_pointer_p
  store i8* %__overflow_area_pointer.next, i8** %__current_saved_reg_area_pointer_p
  %6 = bitcast i8* %__overflow_area_pointer to i32*
  br label %vaarg.end

vaarg.end:                                        ; preds = %vaarg.on_stack, %vaarg.in_reg
  %vaarg.addr = phi i32* [ %5, %vaarg.in_reg ], [ %6, %vaarg.on_stack ]
  %7 = load i32, i32* %vaarg.addr
  store i32 %7, i32* %d, align 4
  %8 = load i32, i32* %d, align 4
  %9 = load i32, i32* %ret, align 4
  %add5 = add nsw i32 %9, %8
  store i32 %add5, i32* %ret, align 4
  %arraydecay6 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0
  %__overflow_area_pointer_p7 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %arraydecay6, i32 0, i32 2
  %__overflow_area_pointer8 = load i8*, i8** %__overflow_area_pointer_p7
  %10 = bitcast i8* %__overflow_area_pointer8 to %struct.AAA*
  %__overflow_area_pointer.next9 = getelementptr i8, i8* %__overflow_area_pointer8, i32 16
  store i8* %__overflow_area_pointer.next9, i8** %__overflow_area_pointer_p7
  %11 = bitcast %struct.AAA* %ddd to i8*
  %12 = bitcast %struct.AAA* %10 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %11, i8* %12, i32 16, i32 4, i1 false)
  %d10 = getelementptr inbounds %struct.AAA, %struct.AAA* %ddd, i32 0, i32 3
  %13 = load i32, i32* %d10, align 4
  %14 = load i32, i32* %ret, align 4
  %add11 = add nsw i32 %14, %13
  store i32 %add11, i32* %ret, align 4
  %arraydecay12 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0
  %__overflow_area_pointer_p13 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %arraydecay12, i32 0, i32 2
  %__overflow_area_pointer14 = load i8*, i8** %__overflow_area_pointer_p13
  %15 = bitcast i8* %__overflow_area_pointer14 to %struct.AAA*
  %__overflow_area_pointer.next15 = getelementptr i8, i8* %__overflow_area_pointer14, i32 16
  store i8* %__overflow_area_pointer.next15, i8** %__overflow_area_pointer_p13
  %16 = bitcast %struct.AAA* %ggg to i8*
  %17 = bitcast %struct.AAA* %15 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %16, i8* %17, i32 16, i32 4, i1 false)
  %d16 = getelementptr inbounds %struct.AAA, %struct.AAA* %ggg, i32 0, i32 3
  %18 = load i32, i32* %d16, align 4
  %19 = load i32, i32* %ret, align 4
  %add17 = add nsw i32 %19, %18
  store i32 %add17, i32* %ret, align 4
  %arraydecay18 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0
  %__overflow_area_pointer_p19 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %arraydecay18, i32 0, i32 2
  %__overflow_area_pointer20 = load i8*, i8** %__overflow_area_pointer_p19
  %20 = bitcast i8* %__overflow_area_pointer20 to %struct.AAA*
  %__overflow_area_pointer.next21 = getelementptr i8, i8* %__overflow_area_pointer20, i32 16
  store i8* %__overflow_area_pointer.next21, i8** %__overflow_area_pointer_p19
  %21 = bitcast %struct.AAA* %nnn to i8*
  %22 = bitcast %struct.AAA* %20 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %21, i8* %22, i32 16, i32 4, i1 false)
  %d22 = getelementptr inbounds %struct.AAA, %struct.AAA* %nnn, i32 0, i32 3
  %23 = load i32, i32* %d22, align 4
  %24 = load i32, i32* %ret, align 4
  %add23 = add nsw i32 %24, %23
  store i32 %add23, i32* %ret, align 4
  %arraydecay24 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0
  br label %vaarg.maybe_reg25

vaarg.maybe_reg25:                                ; preds = %vaarg.end
  %__current_saved_reg_area_pointer_p26 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %arraydecay24, i32 0, i32 0
  %__current_saved_reg_area_pointer27 = load i8*, i8** %__current_saved_reg_area_pointer_p26
  %__saved_reg_area_end_pointer_p28 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %arraydecay24, i32 0, i32 1
  %__saved_reg_area_end_pointer29 = load i8*, i8** %__saved_reg_area_end_pointer_p28
  %__new_saved_reg_area_pointer30 = getelementptr i8, i8* %__current_saved_reg_area_pointer27, i32 4
  %25 = icmp sgt i8* %__new_saved_reg_area_pointer30, %__saved_reg_area_end_pointer29
  br i1 %25, label %vaarg.on_stack32, label %vaarg.in_reg31

vaarg.in_reg31:                                   ; preds = %vaarg.maybe_reg25
  %26 = bitcast i8* %__current_saved_reg_area_pointer27 to i32*
  store i8* %__new_saved_reg_area_pointer30, i8** %__current_saved_reg_area_pointer_p26
  br label %vaarg.end36

vaarg.on_stack32:                                 ; preds = %vaarg.maybe_reg25
  %__overflow_area_pointer_p33 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %arraydecay24, i32 0, i32 2
  %__overflow_area_pointer34 = load i8*, i8** %__overflow_area_pointer_p33
  %__overflow_area_pointer.next35 = getelementptr i8, i8* %__overflow_area_pointer34, i32 4
  store i8* %__overflow_area_pointer.next35, i8** %__overflow_area_pointer_p33
  store i8* %__overflow_area_pointer.next35, i8** %__current_saved_reg_area_pointer_p26
  %27 = bitcast i8* %__overflow_area_pointer34 to i32*
  br label %vaarg.end36

vaarg.end36:                                      ; preds = %vaarg.on_stack32, %vaarg.in_reg31
  %vaarg.addr37 = phi i32* [ %26, %vaarg.in_reg31 ], [ %27, %vaarg.on_stack32 ]
  %28 = load i32, i32* %vaarg.addr37
  store i32 %28, i32* %d, align 4
  %29 = load i32, i32* %d, align 4
  %30 = load i32, i32* %ret, align 4
  %add38 = add nsw i32 %30, %29
  store i32 %add38, i32* %ret, align 4
  %31 = load i32, i32* %m.addr, align 4
  %32 = load i32, i32* %ret, align 4
  %add39 = add nsw i32 %32, %31
  store i32 %add39, i32* %ret, align 4
  %arraydecay40 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0
  %arraydecay4041 = bitcast %struct.__va_list_tag* %arraydecay40 to i8*
  call void @llvm.va_end(i8* %arraydecay4041)
  %33 = load i32, i32* %ret, align 4
  ret i32 %33
}

; Function Attrs: nounwind
declare void @llvm.va_start(i8*) #1

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i32, i1) #1

; Function Attrs: nounwind
declare void @llvm.va_end(i8*) #1

; Function Attrs: nounwind
define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %x = alloca i32, align 4
  store i32 0, i32* %retval
  %call = call i32 (i32, i32, i32, %struct.AAA*, %struct.AAA*, ...) @foo(i32 1, i32 3, i32 5, %struct.AAA* byval(%struct.AAA) align 4 @aaa, %struct.AAA* byval(%struct.AAA) align 4 @fff, i32 2, %struct.AAA* byval(%struct.AAA) align 4 @xxx, %struct.AAA* byval(%struct.AAA) align 4 @yyy, %struct.AAA* byval(%struct.AAA) align 4 @ccc, i32 4)
  store i32 %call, i32* %x, align 4
  %0 = load i32, i32* %x, align 4
  %call1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str, i32 0, i32 0), i32 %0)
  %1 = load i32, i32* %x, align 4
  ret i32 %1
}

declare i32 @printf(i8*, ...) #2

attributes #0 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"Clang 3.1"}
