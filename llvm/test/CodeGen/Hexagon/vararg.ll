; RUN: llc -mtriple=hexagon -mcpu=hexagonv62 -mtriple=hexagon-unknown-linux-musl -O0 < %s | FileCheck %s

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
%struct.__va_list_tag = type { ptr, ptr, ptr }

@aaa = global %struct.AAA { i32 100, i32 200, i32 300, i32 400 }, align 4
@.str = private unnamed_addr constant [13 x i8] c"result = %d\0A\00", align 1

; Function Attrs: nounwind
define i32 @foo(i32 %xx, ...) #0 {
entry:
  %ap = alloca [1 x %struct.__va_list_tag], align 8
  call void @llvm.va_start(ptr %ap)
  %__current_saved_reg_area_pointer = load ptr, ptr %ap, align 8
  %__saved_reg_area_end_pointer_p = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %ap, i32 0, i32 0, i32 1
  %__saved_reg_area_end_pointer = load ptr, ptr %__saved_reg_area_end_pointer_p, align 4
  %__new_saved_reg_area_pointer = getelementptr i8, ptr %__current_saved_reg_area_pointer, i32 4
  %0 = icmp sgt ptr %__new_saved_reg_area_pointer, %__saved_reg_area_end_pointer
  %__overflow_area_pointer_p = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %ap, i32 0, i32 0, i32 2
  %__overflow_area_pointer = load ptr, ptr %__overflow_area_pointer_p, align 8
  br i1 %0, label %vaarg.on_stack, label %vaarg.end

vaarg.on_stack:                                   ; preds = %entry
  %__overflow_area_pointer.next = getelementptr i8, ptr %__overflow_area_pointer, i32 4
  store ptr %__overflow_area_pointer.next, ptr %__overflow_area_pointer_p, align 8
  br label %vaarg.end

vaarg.end:                                        ; preds = %entry, %vaarg.on_stack
  %__overflow_area_pointer5 = phi ptr [ %__overflow_area_pointer.next, %vaarg.on_stack ], [ %__overflow_area_pointer, %entry ]
  %storemerge32 = phi ptr [ %__overflow_area_pointer.next, %vaarg.on_stack ], [ %__new_saved_reg_area_pointer, %entry ]
  %vaarg.addr.in = phi ptr [ %__overflow_area_pointer, %vaarg.on_stack ], [ %__current_saved_reg_area_pointer, %entry ]
  store ptr %storemerge32, ptr %ap, align 8
  %1 = load i32, ptr %vaarg.addr.in, align 4
  %__overflow_area_pointer_p4 = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %ap, i32 0, i32 0, i32 2
  %__overflow_area_pointer.next6 = getelementptr i8, ptr %__overflow_area_pointer5, i32 16
  store ptr %__overflow_area_pointer.next6, ptr %__overflow_area_pointer_p4, align 8
  %bbb.sroa.1.0.idx27 = getelementptr inbounds i8, ptr %__overflow_area_pointer5, i32 12
  %bbb.sroa.1.0.copyload = load i32, ptr %bbb.sroa.1.0.idx27, align 4
  %add8 = add nsw i32 %bbb.sroa.1.0.copyload, %1
  %__new_saved_reg_area_pointer15 = getelementptr i8, ptr %storemerge32, i32 4
  %2 = icmp sgt ptr %__new_saved_reg_area_pointer15, %__saved_reg_area_end_pointer
  br i1 %2, label %vaarg.on_stack17, label %vaarg.end21

vaarg.on_stack17:                                 ; preds = %vaarg.end
  %__overflow_area_pointer.next20 = getelementptr i8, ptr %__overflow_area_pointer5, i32 20
  store ptr %__overflow_area_pointer.next20, ptr %__overflow_area_pointer_p4, align 8
  br label %vaarg.end21

vaarg.end21:                                      ; preds = %vaarg.end, %vaarg.on_stack17
  %storemerge = phi ptr [ %__overflow_area_pointer.next20, %vaarg.on_stack17 ], [ %__new_saved_reg_area_pointer15, %vaarg.end ]
  %vaarg.addr22.in = phi ptr [ %__overflow_area_pointer.next6, %vaarg.on_stack17 ], [ %storemerge32, %vaarg.end ]
  store ptr %storemerge, ptr %ap, align 8
  %3 = load i32, ptr %vaarg.addr22.in, align 4
  %add23 = add nsw i32 %add8, %3
  call void @llvm.va_end(ptr %ap)
  ret i32 %add23
}

; Function Attrs: nounwind
declare void @llvm.va_start(ptr) #1

; Function Attrs: nounwind
declare void @llvm.va_end(ptr) #1

; Function Attrs: nounwind
define i32 @main() #0 {
entry:
  %call = tail call i32 (i32, ...) @foo(i32 undef, i32 2, ptr byval(%struct.AAA) align 4 @aaa, i32 4)
  %call1 = tail call i32 (ptr, ...) @printf(ptr @.str, i32 %call) #1
  ret i32 %call
}

; Function Attrs: nounwind
declare i32 @printf(ptr nocapture readonly, ...) #0

attributes #0 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"Clang 3.1"}
