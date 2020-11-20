; RUN: llc -march=hexagon -mcpu=hexagonv62  -mtriple=hexagon-unknown-linux-musl -O0 < %s | FileCheck %s

; CHECK-LABEL: foo:

; Check Function prologue.
; Note. All register numbers and offset are fixed.
; Hence, no need of regular expression.

; CHECK: r29 = add(r29,#-8)
; CHECK: memw(r29+#4) = r5
; CHECK: r29 = add(r29,#8)

%struct.AAA = type { i32, i32, i32, i32 }
%struct.__va_list_tag = type { i8*, i8*, i8* }

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
  store i32 %xx, i32* %xx.addr, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 %b, i32* %b.addr, align 4
  store i32 %c, i32* %c.addr, align 4
  store i32 %x, i32* %x.addr, align 4
  store i32 0, i32* %ret, align 4
  %arraydecay = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0
  %arraydecay1 = bitcast %struct.__va_list_tag* %arraydecay to i8*
  call void @llvm.va_start(i8* %arraydecay1)
  %arraydecay2 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0
  br label %vaarg.maybe_reg

vaarg.maybe_reg:                                  ; preds = %entry
  %__current_saved_reg_area_pointer_p = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %arraydecay2, i32 0, i32 0
  %__current_saved_reg_area_pointer = load i8*, i8** %__current_saved_reg_area_pointer_p
  %__saved_reg_area_end_pointer_p = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %arraydecay2, i32 0, i32 1
  %__saved_reg_area_end_pointer = load i8*, i8** %__saved_reg_area_end_pointer_p
  %0 = ptrtoint i8* %__current_saved_reg_area_pointer to i32
  %align_current_saved_reg_area_pointer = add i32 %0, 7
  %align_current_saved_reg_area_pointer3 = and i32 %align_current_saved_reg_area_pointer, -8
  %align_current_saved_reg_area_pointer4 = inttoptr i32 %align_current_saved_reg_area_pointer3 to i8*
  %__new_saved_reg_area_pointer = getelementptr i8, i8* %align_current_saved_reg_area_pointer4, i32 8
  %1 = icmp sgt i8* %__new_saved_reg_area_pointer, %__saved_reg_area_end_pointer
  br i1 %1, label %vaarg.on_stack, label %vaarg.in_reg

vaarg.in_reg:                                     ; preds = %vaarg.maybe_reg
  %2 = bitcast i8* %align_current_saved_reg_area_pointer4 to i64*
  store i8* %__new_saved_reg_area_pointer, i8** %__current_saved_reg_area_pointer_p
  br label %vaarg.end

vaarg.on_stack:                                   ; preds = %vaarg.maybe_reg
  %__overflow_area_pointer_p = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %arraydecay2, i32 0, i32 2
  %__overflow_area_pointer = load i8*, i8** %__overflow_area_pointer_p
  %3 = ptrtoint i8* %__overflow_area_pointer to i32
  %align_overflow_area_pointer = add i32 %3, 7
  %align_overflow_area_pointer5 = and i32 %align_overflow_area_pointer, -8
  %align_overflow_area_pointer6 = inttoptr i32 %align_overflow_area_pointer5 to i8*
  %__overflow_area_pointer.next = getelementptr i8, i8* %align_overflow_area_pointer6, i32 8
  store i8* %__overflow_area_pointer.next, i8** %__overflow_area_pointer_p
  store i8* %__overflow_area_pointer.next, i8** %__current_saved_reg_area_pointer_p
  %4 = bitcast i8* %align_overflow_area_pointer6 to i64*
  br label %vaarg.end

vaarg.end:                                        ; preds = %vaarg.on_stack, %vaarg.in_reg
  %vaarg.addr = phi i64* [ %2, %vaarg.in_reg ], [ %4, %vaarg.on_stack ]
  %5 = load i64, i64* %vaarg.addr
  %conv = trunc i64 %5 to i32
  store i32 %conv, i32* %d, align 4
  %6 = load i32, i32* %d, align 4
  %7 = load i32, i32* %ret, align 4
  %add = add nsw i32 %7, %6
  store i32 %add, i32* %ret, align 4
  %arraydecay7 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0
  %__overflow_area_pointer_p8 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %arraydecay7, i32 0, i32 2
  %__overflow_area_pointer9 = load i8*, i8** %__overflow_area_pointer_p8
  %8 = bitcast i8* %__overflow_area_pointer9 to %struct.AAA*
  %__overflow_area_pointer.next10 = getelementptr i8, i8* %__overflow_area_pointer9, i32 16
  store i8* %__overflow_area_pointer.next10, i8** %__overflow_area_pointer_p8
  %9 = bitcast %struct.AAA* %bbb to i8*
  %10 = bitcast %struct.AAA* %8 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %9, i8* %10, i32 16, i32 4, i1 false)
  %d11 = getelementptr inbounds %struct.AAA, %struct.AAA* %bbb, i32 0, i32 3
  %11 = load i32, i32* %d11, align 4
  %12 = load i32, i32* %ret, align 4
  %add12 = add nsw i32 %12, %11
  store i32 %add12, i32* %ret, align 4
  %arraydecay13 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0
  br label %vaarg.maybe_reg14

vaarg.maybe_reg14:                                ; preds = %vaarg.end
  %__current_saved_reg_area_pointer_p15 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %arraydecay13, i32 0, i32 0
  %__current_saved_reg_area_pointer16 = load i8*, i8** %__current_saved_reg_area_pointer_p15
  %__saved_reg_area_end_pointer_p17 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %arraydecay13, i32 0, i32 1
  %__saved_reg_area_end_pointer18 = load i8*, i8** %__saved_reg_area_end_pointer_p17
  %__new_saved_reg_area_pointer19 = getelementptr i8, i8* %__current_saved_reg_area_pointer16, i32 4
  %13 = icmp sgt i8* %__new_saved_reg_area_pointer19, %__saved_reg_area_end_pointer18
  br i1 %13, label %vaarg.on_stack21, label %vaarg.in_reg20

vaarg.in_reg20:                                   ; preds = %vaarg.maybe_reg14
  %14 = bitcast i8* %__current_saved_reg_area_pointer16 to i32*
  store i8* %__new_saved_reg_area_pointer19, i8** %__current_saved_reg_area_pointer_p15
  br label %vaarg.end25

vaarg.on_stack21:                                 ; preds = %vaarg.maybe_reg14
  %__overflow_area_pointer_p22 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %arraydecay13, i32 0, i32 2
  %__overflow_area_pointer23 = load i8*, i8** %__overflow_area_pointer_p22
  %__overflow_area_pointer.next24 = getelementptr i8, i8* %__overflow_area_pointer23, i32 4
  store i8* %__overflow_area_pointer.next24, i8** %__overflow_area_pointer_p22
  store i8* %__overflow_area_pointer.next24, i8** %__current_saved_reg_area_pointer_p15
  %15 = bitcast i8* %__overflow_area_pointer23 to i32*
  br label %vaarg.end25

vaarg.end25:                                      ; preds = %vaarg.on_stack21, %vaarg.in_reg20
  %vaarg.addr26 = phi i32* [ %14, %vaarg.in_reg20 ], [ %15, %vaarg.on_stack21 ]
  %16 = load i32, i32* %vaarg.addr26
  store i32 %16, i32* %d, align 4
  %17 = load i32, i32* %d, align 4
  %18 = load i32, i32* %ret, align 4
  %add27 = add nsw i32 %18, %17
  store i32 %add27, i32* %ret, align 4
  %arraydecay28 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0
  br label %vaarg.maybe_reg29

vaarg.maybe_reg29:                                ; preds = %vaarg.end25
  %__current_saved_reg_area_pointer_p30 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %arraydecay28, i32 0, i32 0
  %__current_saved_reg_area_pointer31 = load i8*, i8** %__current_saved_reg_area_pointer_p30
  %__saved_reg_area_end_pointer_p32 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %arraydecay28, i32 0, i32 1
  %__saved_reg_area_end_pointer33 = load i8*, i8** %__saved_reg_area_end_pointer_p32
  %19 = ptrtoint i8* %__current_saved_reg_area_pointer31 to i32
  %align_current_saved_reg_area_pointer34 = add i32 %19, 7
  %align_current_saved_reg_area_pointer35 = and i32 %align_current_saved_reg_area_pointer34, -8
  %align_current_saved_reg_area_pointer36 = inttoptr i32 %align_current_saved_reg_area_pointer35 to i8*
  %__new_saved_reg_area_pointer37 = getelementptr i8, i8* %align_current_saved_reg_area_pointer36, i32 8
  %20 = icmp sgt i8* %__new_saved_reg_area_pointer37, %__saved_reg_area_end_pointer33
  br i1 %20, label %vaarg.on_stack39, label %vaarg.in_reg38

vaarg.in_reg38:                                   ; preds = %vaarg.maybe_reg29
  %21 = bitcast i8* %align_current_saved_reg_area_pointer36 to i64*
  store i8* %__new_saved_reg_area_pointer37, i8** %__current_saved_reg_area_pointer_p30
  br label %vaarg.end46

vaarg.on_stack39:                                 ; preds = %vaarg.maybe_reg29
  %__overflow_area_pointer_p40 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %arraydecay28, i32 0, i32 2
  %__overflow_area_pointer41 = load i8*, i8** %__overflow_area_pointer_p40
  %22 = ptrtoint i8* %__overflow_area_pointer41 to i32
  %align_overflow_area_pointer42 = add i32 %22, 7
  %align_overflow_area_pointer43 = and i32 %align_overflow_area_pointer42, -8
  %align_overflow_area_pointer44 = inttoptr i32 %align_overflow_area_pointer43 to i8*
  %__overflow_area_pointer.next45 = getelementptr i8, i8* %align_overflow_area_pointer44, i32 8
  store i8* %__overflow_area_pointer.next45, i8** %__overflow_area_pointer_p40
  store i8* %__overflow_area_pointer.next45, i8** %__current_saved_reg_area_pointer_p30
  %23 = bitcast i8* %align_overflow_area_pointer44 to i64*
  br label %vaarg.end46

vaarg.end46:                                      ; preds = %vaarg.on_stack39, %vaarg.in_reg38
  %vaarg.addr47 = phi i64* [ %21, %vaarg.in_reg38 ], [ %23, %vaarg.on_stack39 ]
  %24 = load i64, i64* %vaarg.addr47
  %conv48 = trunc i64 %24 to i32
  store i32 %conv48, i32* %d, align 4
  %25 = load i32, i32* %d, align 4
  %26 = load i32, i32* %ret, align 4
  %add49 = add nsw i32 %26, %25
  store i32 %add49, i32* %ret, align 4
  %arraydecay50 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0
  %arraydecay5051 = bitcast %struct.__va_list_tag* %arraydecay50 to i8*
  call void @llvm.va_end(i8* %arraydecay5051)
  %27 = load i32, i32* %ret, align 4
  ret i32 %27
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
  %y = alloca i64, align 8
  store i32 0, i32* %retval
  store i64 1000000, i64* %y, align 8
  %0 = load i64, i64* %y, align 8
  %1 = load i64, i64* %y, align 8
  %call = call i32 (i32, i32, i32, i32, i32, ...) @foo(i32 1, i32 2, i32 3, i32 4, i32 5, i64 %0, %struct.AAA* byval(%struct.AAA) align 4 @aaa, i32 4, i64 %1)
  store i32 %call, i32* %x, align 4
  %2 = load i32, i32* %x, align 4
  %call1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str, i32 0, i32 0), i32 %2)
  %3 = load i32, i32* %x, align 4
  ret i32 %3
}

declare i32 @printf(i8*, ...) #2

attributes #0 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"Clang 3.1"}
