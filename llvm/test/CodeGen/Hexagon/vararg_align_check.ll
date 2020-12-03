; RUN: llc -march=hexagon -mcpu=hexagonv62  -mtriple=hexagon-unknown-linux-musl -O0 < %s | FileCheck %s

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
%struct.__va_list_tag = type { i8*, i8*, i8* }

@aaa = global %struct.AAA { i32 100, i32 200, i32 300, i32 400 }, align 4
@ddd = global { i8, i64, i32, [4 x i8] } { i8 1, i64 1000000, i32 5, [4 x i8] undef }, align 8
@.str = private unnamed_addr constant [13 x i8] c"result = %d\0A\00", align 1

; Function Attrs: nounwind
define i32 @foo(i32 %xx, %struct.BBB* byval(%struct.BBB) align 8 %eee, ...) #0 {
entry:
  %xx.addr = alloca i32, align 4
  %ap = alloca [1 x %struct.__va_list_tag], align 8
  %d = alloca i32, align 4
  %k = alloca i64, align 8
  %ret = alloca i32, align 4
  %bbb = alloca %struct.AAA, align 4
  store i32 %xx, i32* %xx.addr, align 4
  store i32 0, i32* %ret, align 4
  %x = getelementptr inbounds %struct.BBB, %struct.BBB* %eee, i32 0, i32 0
  %0 = load i8, i8* %x, align 1
  %tobool = trunc i8 %0 to i1
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 1, i32* %ret, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %arraydecay = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0
  %arraydecay1 = bitcast %struct.__va_list_tag* %arraydecay to i8*
  call void @llvm.va_start(i8* %arraydecay1)
  %arraydecay2 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0
  br label %vaarg.maybe_reg

vaarg.maybe_reg:                                  ; preds = %if.end
  %__current_saved_reg_area_pointer_p = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %arraydecay2, i32 0, i32 0
  %__current_saved_reg_area_pointer = load i8*, i8** %__current_saved_reg_area_pointer_p
  %__saved_reg_area_end_pointer_p = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %arraydecay2, i32 0, i32 1
  %__saved_reg_area_end_pointer = load i8*, i8** %__saved_reg_area_end_pointer_p
  %1 = ptrtoint i8* %__current_saved_reg_area_pointer to i32
  %align_current_saved_reg_area_pointer = add i32 %1, 7
  %align_current_saved_reg_area_pointer3 = and i32 %align_current_saved_reg_area_pointer, -8
  %align_current_saved_reg_area_pointer4 = inttoptr i32 %align_current_saved_reg_area_pointer3 to i8*
  %__new_saved_reg_area_pointer = getelementptr i8, i8* %align_current_saved_reg_area_pointer4, i32 8
  %2 = icmp sgt i8* %__new_saved_reg_area_pointer, %__saved_reg_area_end_pointer
  br i1 %2, label %vaarg.on_stack, label %vaarg.in_reg

vaarg.in_reg:                                     ; preds = %vaarg.maybe_reg
  %3 = bitcast i8* %align_current_saved_reg_area_pointer4 to i64*
  store i8* %__new_saved_reg_area_pointer, i8** %__current_saved_reg_area_pointer_p
  br label %vaarg.end

vaarg.on_stack:                                   ; preds = %vaarg.maybe_reg
  %__overflow_area_pointer_p = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %arraydecay2, i32 0, i32 2
  %__overflow_area_pointer = load i8*, i8** %__overflow_area_pointer_p
  %4 = ptrtoint i8* %__overflow_area_pointer to i32
  %align_overflow_area_pointer = add i32 %4, 7
  %align_overflow_area_pointer5 = and i32 %align_overflow_area_pointer, -8
  %align_overflow_area_pointer6 = inttoptr i32 %align_overflow_area_pointer5 to i8*
  %__overflow_area_pointer.next = getelementptr i8, i8* %align_overflow_area_pointer6, i32 8
  store i8* %__overflow_area_pointer.next, i8** %__overflow_area_pointer_p
  store i8* %__overflow_area_pointer.next, i8** %__current_saved_reg_area_pointer_p
  %5 = bitcast i8* %align_overflow_area_pointer6 to i64*
  br label %vaarg.end

vaarg.end:                                        ; preds = %vaarg.on_stack, %vaarg.in_reg
  %vaarg.addr = phi i64* [ %3, %vaarg.in_reg ], [ %5, %vaarg.on_stack ]
  %6 = load i64, i64* %vaarg.addr
  store i64 %6, i64* %k, align 8
  %7 = load i64, i64* %k, align 8
  %conv = trunc i64 %7 to i32
  %div = sdiv i32 %conv, 1000
  %8 = load i32, i32* %ret, align 4
  %add = add nsw i32 %8, %div
  store i32 %add, i32* %ret, align 4
  %arraydecay7 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0
  %__overflow_area_pointer_p8 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %arraydecay7, i32 0, i32 2
  %__overflow_area_pointer9 = load i8*, i8** %__overflow_area_pointer_p8
  %9 = bitcast i8* %__overflow_area_pointer9 to %struct.AAA*
  %__overflow_area_pointer.next10 = getelementptr i8, i8* %__overflow_area_pointer9, i32 16
  store i8* %__overflow_area_pointer.next10, i8** %__overflow_area_pointer_p8
  %10 = bitcast %struct.AAA* %bbb to i8*
  %11 = bitcast %struct.AAA* %9 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %10, i8* %11, i32 16, i32 4, i1 false)
  %d11 = getelementptr inbounds %struct.AAA, %struct.AAA* %bbb, i32 0, i32 3
  %12 = load i32, i32* %d11, align 4
  %13 = load i32, i32* %ret, align 4
  %add12 = add nsw i32 %13, %12
  store i32 %add12, i32* %ret, align 4
  %arraydecay13 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0
  br label %vaarg.maybe_reg14

vaarg.maybe_reg14:                                ; preds = %vaarg.end
  %__current_saved_reg_area_pointer_p15 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %arraydecay13, i32 0, i32 0
  %__current_saved_reg_area_pointer16 = load i8*, i8** %__current_saved_reg_area_pointer_p15
  %__saved_reg_area_end_pointer_p17 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %arraydecay13, i32 0, i32 1
  %__saved_reg_area_end_pointer18 = load i8*, i8** %__saved_reg_area_end_pointer_p17
  %__new_saved_reg_area_pointer19 = getelementptr i8, i8* %__current_saved_reg_area_pointer16, i32 4
  %14 = icmp sgt i8* %__new_saved_reg_area_pointer19, %__saved_reg_area_end_pointer18
  br i1 %14, label %vaarg.on_stack21, label %vaarg.in_reg20

vaarg.in_reg20:                                   ; preds = %vaarg.maybe_reg14
  %15 = bitcast i8* %__current_saved_reg_area_pointer16 to i32*
  store i8* %__new_saved_reg_area_pointer19, i8** %__current_saved_reg_area_pointer_p15
  br label %vaarg.end25

vaarg.on_stack21:                                 ; preds = %vaarg.maybe_reg14
  %__overflow_area_pointer_p22 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %arraydecay13, i32 0, i32 2
  %__overflow_area_pointer23 = load i8*, i8** %__overflow_area_pointer_p22
  %__overflow_area_pointer.next24 = getelementptr i8, i8* %__overflow_area_pointer23, i32 4
  store i8* %__overflow_area_pointer.next24, i8** %__overflow_area_pointer_p22
  store i8* %__overflow_area_pointer.next24, i8** %__current_saved_reg_area_pointer_p15
  %16 = bitcast i8* %__overflow_area_pointer23 to i32*
  br label %vaarg.end25

vaarg.end25:                                      ; preds = %vaarg.on_stack21, %vaarg.in_reg20
  %vaarg.addr26 = phi i32* [ %15, %vaarg.in_reg20 ], [ %16, %vaarg.on_stack21 ]
  %17 = load i32, i32* %vaarg.addr26
  store i32 %17, i32* %d, align 4
  %18 = load i32, i32* %d, align 4
  %19 = load i32, i32* %ret, align 4
  %add27 = add nsw i32 %19, %18
  store i32 %add27, i32* %ret, align 4
  %arraydecay28 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0
  %arraydecay2829 = bitcast %struct.__va_list_tag* %arraydecay28 to i8*
  call void @llvm.va_end(i8* %arraydecay2829)
  %20 = load i32, i32* %ret, align 4
  ret i32 %20
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
  %m = alloca i64, align 8
  store i32 0, i32* %retval
  store i64 1000000, i64* %m, align 8
  %0 = load i64, i64* %m, align 8
  %call = call i32 (i32, %struct.BBB*, ...) @foo(i32 1, %struct.BBB* byval(%struct.BBB) align 8 bitcast ({ i8, i64, i32, [4 x i8] }* @ddd to %struct.BBB*), i64 %0, %struct.AAA* byval(%struct.AAA) align 4 @aaa, i32 4)
  store i32 %call, i32* %x, align 4
  %1 = load i32, i32* %x, align 4
  %call1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str, i32 0, i32 0), i32 %1)
  %2 = load i32, i32* %x, align 4
  ret i32 %2
}

declare i32 @printf(i8*, ...) #2

attributes #1 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"Clang 3.1"}
