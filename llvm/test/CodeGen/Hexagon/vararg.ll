; RUN: llc -march=hexagon -mcpu=hexagonv62 -mtriple=hexagon-unknown-linux-musl  -O0 < %s | FileCheck %s

; CHECK-LABEL: foo:

; Check function prologue generation
; CHECK: r29 = add(r29,#-24)
; CHECK: memw(r29+#4) = r1
; CHECK: memw(r29+#8) = r2
; CHECK: memw(r29+#12) = r3
; CHECK: memw(r29+#16) = r4
; CHECK: memw(r29+#20) = r5
; CHECK: r29 = add(r29,#24)


%struct.AAA = type { i32, i32, i32, i32 }
%struct.__va_list_tag = type { i8*, i8*, i8* }

@aaa = global %struct.AAA { i32 100, i32 200, i32 300, i32 400 }, align 4
@.str = private unnamed_addr constant [13 x i8] c"result = %d\0A\00", align 1

; Function Attrs: nounwind
define i32 @foo(i32 %xx, ...) #0 {
entry:
  %ap = alloca [1 x %struct.__va_list_tag], align 8
  %arraydecay1 = bitcast [1 x %struct.__va_list_tag]* %ap to i8*
  call void @llvm.va_start(i8* %arraydecay1)
  %__current_saved_reg_area_pointer_p = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0, i32 0
  %__current_saved_reg_area_pointer = load i8*, i8** %__current_saved_reg_area_pointer_p, align 8
  %__saved_reg_area_end_pointer_p = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0, i32 1
  %__saved_reg_area_end_pointer = load i8*, i8** %__saved_reg_area_end_pointer_p, align 4
  %__new_saved_reg_area_pointer = getelementptr i8, i8* %__current_saved_reg_area_pointer, i32 4
  %0 = icmp sgt i8* %__new_saved_reg_area_pointer, %__saved_reg_area_end_pointer
  %__overflow_area_pointer_p = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0, i32 2
  %__overflow_area_pointer = load i8*, i8** %__overflow_area_pointer_p, align 8
  br i1 %0, label %vaarg.on_stack, label %vaarg.end

vaarg.on_stack:                                   ; preds = %entry
  %__overflow_area_pointer.next = getelementptr i8, i8* %__overflow_area_pointer, i32 4
  store i8* %__overflow_area_pointer.next, i8** %__overflow_area_pointer_p, align 8
  br label %vaarg.end

vaarg.end:                                        ; preds = %entry, %vaarg.on_stack
  %__overflow_area_pointer5 = phi i8* [ %__overflow_area_pointer.next, %vaarg.on_stack ], [ %__overflow_area_pointer, %entry ]
  %storemerge32 = phi i8* [ %__overflow_area_pointer.next, %vaarg.on_stack ], [ %__new_saved_reg_area_pointer, %entry ]
  %vaarg.addr.in = phi i8* [ %__overflow_area_pointer, %vaarg.on_stack ], [ %__current_saved_reg_area_pointer, %entry ]
  store i8* %storemerge32, i8** %__current_saved_reg_area_pointer_p, align 8
  %vaarg.addr = bitcast i8* %vaarg.addr.in to i32*
  %1 = load i32, i32* %vaarg.addr, align 4
  %__overflow_area_pointer_p4 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0, i32 2
  %__overflow_area_pointer.next6 = getelementptr i8, i8* %__overflow_area_pointer5, i32 16
  store i8* %__overflow_area_pointer.next6, i8** %__overflow_area_pointer_p4, align 8
  %bbb.sroa.1.0.idx27 = getelementptr inbounds i8, i8* %__overflow_area_pointer5, i32 12
  %2 = bitcast i8* %bbb.sroa.1.0.idx27 to i32*
  %bbb.sroa.1.0.copyload = load i32, i32* %2, align 4
  %add8 = add nsw i32 %bbb.sroa.1.0.copyload, %1
  %__new_saved_reg_area_pointer15 = getelementptr i8, i8* %storemerge32, i32 4
  %3 = icmp sgt i8* %__new_saved_reg_area_pointer15, %__saved_reg_area_end_pointer
  br i1 %3, label %vaarg.on_stack17, label %vaarg.end21

vaarg.on_stack17:                                 ; preds = %vaarg.end
  %__overflow_area_pointer.next20 = getelementptr i8, i8* %__overflow_area_pointer5, i32 20
  store i8* %__overflow_area_pointer.next20, i8** %__overflow_area_pointer_p4, align 8
  br label %vaarg.end21

vaarg.end21:                                      ; preds = %vaarg.end, %vaarg.on_stack17
  %storemerge = phi i8* [ %__overflow_area_pointer.next20, %vaarg.on_stack17 ], [ %__new_saved_reg_area_pointer15, %vaarg.end ]
  %vaarg.addr22.in = phi i8* [ %__overflow_area_pointer.next6, %vaarg.on_stack17 ], [ %storemerge32, %vaarg.end ]
  store i8* %storemerge, i8** %__current_saved_reg_area_pointer_p, align 8
  %vaarg.addr22 = bitcast i8* %vaarg.addr22.in to i32*
  %4 = load i32, i32* %vaarg.addr22, align 4
  %add23 = add nsw i32 %add8, %4
  call void @llvm.va_end(i8* %arraydecay1)
  ret i32 %add23
}

; Function Attrs: nounwind
declare void @llvm.va_start(i8*) #1

; Function Attrs: nounwind
declare void @llvm.va_end(i8*) #1

; Function Attrs: nounwind
define i32 @main() #0 {
entry:
  %call = tail call i32 (i32, ...) @foo(i32 undef, i32 2, %struct.AAA* byval(%struct.AAA) align 4 @aaa, i32 4)
  %call1 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str, i32 0, i32 0), i32 %call) #1
  ret i32 %call
}

; Function Attrs: nounwind
declare i32 @printf(i8* nocapture readonly, ...) #0

attributes #0 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"Clang 3.1"}
