; RUN: opt < %s -passes=jump-threading -S | FileCheck %s
; Test whether two consecutive switches with identical structures assign the
; proper value to the proper variable.  This is really testing
; Instruction::isIdenticalToWhenDefined, as previously that function was
; returning true if the value part of the operands of two phis were identical,
; even if the incoming blocks were not.
; NB: this function should be pruned down more.

%struct._GList = type { ptr, ptr, ptr }
%struct.filter_def = type { ptr, ptr }

@capture_filters = external hidden global ptr, align 8
@display_filters = external hidden global ptr, align 8
@.str2 = external hidden unnamed_addr constant [10 x i8], align 1
@__PRETTY_FUNCTION__.copy_filter_list = external hidden unnamed_addr constant [62 x i8], align 1
@.str12 = external hidden unnamed_addr constant [22 x i8], align 1
@.str13 = external hidden unnamed_addr constant [31 x i8], align 1
@capture_edited_filters = external hidden global ptr, align 8
@display_edited_filters = external hidden global ptr, align 8
@__PRETTY_FUNCTION__.get_filter_list = external hidden unnamed_addr constant [44 x i8], align 1

declare void @g_assertion_message(ptr, ptr, i32, ptr, ptr) noreturn

declare void @g_free(ptr)

declare ptr @g_list_first(ptr)

declare noalias ptr @g_malloc(i64)

define void @copy_filter_list(i32 %dest_type, i32 %src_type) nounwind uwtable ssp {
entry:
  br label %do.body

do.body:                                          ; preds = %entry
  %cmp = icmp ne i32 %dest_type, %src_type
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %do.body
  br label %if.end

if.else:                                          ; preds = %do.body
  call void @g_assertion_message_expr(ptr null, ptr @.str2, i32 581, ptr @__PRETTY_FUNCTION__.copy_filter_list, ptr @.str12) noreturn
  unreachable

if.end:                                           ; preds = %if.then
  br label %do.end

do.end:                                           ; preds = %if.end
  switch i32 %dest_type, label %sw.default.i [
    i32 0, label %sw.bb.i
    i32 1, label %sw.bb1.i
    i32 2, label %sw.bb2.i
    i32 3, label %sw.bb3.i
  ]

sw.bb.i:                                          ; preds = %do.end
  br label %get_filter_list.exit

sw.bb1.i:                                         ; preds = %do.end
  br label %get_filter_list.exit

sw.bb2.i:                                         ; preds = %do.end
  br label %get_filter_list.exit

sw.bb3.i:                                         ; preds = %do.end
  br label %get_filter_list.exit

sw.default.i:                                     ; preds = %do.end
  call void @g_assertion_message(ptr null, ptr @.str2, i32 408, ptr @__PRETTY_FUNCTION__.get_filter_list, ptr null) noreturn nounwind
  unreachable

get_filter_list.exit:                             ; preds = %sw.bb3.i, %sw.bb2.i, %sw.bb1.i, %sw.bb.i
  %0 = phi ptr [ @display_edited_filters, %sw.bb3.i ], [ @capture_edited_filters, %sw.bb2.i ], [ @display_filters, %sw.bb1.i ], [ @capture_filters, %sw.bb.i ]
  switch i32 %src_type, label %sw.default.i5 [
    i32 0, label %sw.bb.i1
    i32 1, label %sw.bb1.i2
    i32 2, label %sw.bb2.i3
    i32 3, label %sw.bb3.i4
  ]

sw.bb.i1:                                         ; preds = %get_filter_list.exit
  br label %get_filter_list.exit6

sw.bb1.i2:                                        ; preds = %get_filter_list.exit
  br label %get_filter_list.exit6

sw.bb2.i3:                                        ; preds = %get_filter_list.exit
  br label %get_filter_list.exit6

sw.bb3.i4:                                        ; preds = %get_filter_list.exit
  br label %get_filter_list.exit6

sw.default.i5:                                    ; preds = %get_filter_list.exit
  call void @g_assertion_message(ptr null, ptr @.str2, i32 408, ptr @__PRETTY_FUNCTION__.get_filter_list, ptr null) noreturn nounwind
  unreachable

; CHECK: get_filter_list.exit
get_filter_list.exit6:                            ; preds = %sw.bb3.i4, %sw.bb2.i3, %sw.bb1.i2, %sw.bb.i1
  %1 = phi ptr [ @display_edited_filters, %sw.bb3.i4 ], [ @capture_edited_filters, %sw.bb2.i3 ], [ @display_filters, %sw.bb1.i2 ], [ @capture_filters, %sw.bb.i1 ]
; CHECK: %2 = load
  %2 = load ptr, ptr %1, align 8
; We should have jump-threading insert an additional load here for the value
; coming out of the first switch, which is picked up by a subsequent phi
; CHECK: %.pr = load ptr, ptr %0
; CHECK-NEXT:  br label %while.cond
  br label %while.cond

; CHECK: while.cond
while.cond:                                       ; preds = %while.body, %get_filter_list.exit6
; CHECK: {{= phi .*%.pr}}
  %3 = load ptr, ptr %0, align 8
; CHECK: tobool
  %tobool = icmp ne ptr %3, null
  br i1 %tobool, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %4 = load ptr, ptr %0, align 8
  %5 = load ptr, ptr %0, align 8
  %call2 = call ptr @g_list_first(ptr %5)
  %6 = load ptr, ptr %call2, align 8
  %7 = load ptr, ptr %6, align 8
  call void @g_free(ptr %7) nounwind
  %strval.i = getelementptr inbounds %struct.filter_def, ptr %6, i32 0, i32 1
  %8 = load ptr, ptr %strval.i, align 8
  call void @g_free(ptr %8) nounwind
  call void @g_free(ptr %6) nounwind
  %call.i = call ptr @g_list_remove_link(ptr %4, ptr %call2) nounwind
  store ptr %call.i, ptr %0, align 8
  br label %while.cond

while.end:                                        ; preds = %while.cond
  br label %do.body4

do.body4:                                         ; preds = %while.end
  %9 = load ptr, ptr %0, align 8
  %call5 = call i32 @g_list_length(ptr %9)
  %cmp6 = icmp eq i32 %call5, 0
  br i1 %cmp6, label %if.then7, label %if.else8

if.then7:                                         ; preds = %do.body4
  br label %if.end9

if.else8:                                         ; preds = %do.body4
  call void @g_assertion_message_expr(ptr null, ptr @.str2, i32 600, ptr @__PRETTY_FUNCTION__.copy_filter_list, ptr @.str13) noreturn
  unreachable

if.end9:                                          ; preds = %if.then7
  br label %do.end10

do.end10:                                         ; preds = %if.end9
  br label %while.cond11

while.cond11:                                     ; preds = %cond.end, %do.end10
  %cond10 = phi ptr [ %cond, %cond.end ], [ %2, %do.end10 ]
  %tobool12 = icmp ne ptr %cond10, null
  br i1 %tobool12, label %while.body13, label %while.end16

while.body13:                                     ; preds = %while.cond11
  %10 = load ptr, ptr %cond10, align 8
  %11 = load ptr, ptr %0, align 8
  %12 = load ptr, ptr %10, align 8
  %strval = getelementptr inbounds %struct.filter_def, ptr %10, i32 0, i32 1
  %13 = load ptr, ptr %strval, align 8
  %call.i7 = call noalias ptr @g_malloc(i64 16) nounwind
  %call1.i = call noalias ptr @g_strdup(ptr %12) nounwind
  store ptr %call1.i, ptr %call.i7, align 8
  %call2.i = call noalias ptr @g_strdup(ptr %13) nounwind
  %strval.i9 = getelementptr inbounds %struct.filter_def, ptr %call.i7, i32 0, i32 1
  store ptr %call2.i, ptr %strval.i9, align 8
  %call3.i = call ptr @g_list_append(ptr %11, ptr %call.i7) nounwind
  store ptr %call3.i, ptr %0, align 8
  %tobool15 = icmp ne ptr %cond10, null
  br i1 %tobool15, label %cond.true, label %cond.false

cond.true:                                        ; preds = %while.body13
  %next = getelementptr inbounds %struct._GList, ptr %cond10, i32 0, i32 1
  %14 = load ptr, ptr %next, align 8
  br label %cond.end

cond.false:                                       ; preds = %while.body13
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi ptr [ %14, %cond.true ], [ null, %cond.false ]
  br label %while.cond11

while.end16:                                      ; preds = %while.cond11
  ret void
}

declare void @g_assertion_message_expr(ptr, ptr, i32, ptr, ptr) noreturn

declare i32 @g_list_length(ptr)

declare noalias ptr @g_strdup(ptr)

declare ptr @g_list_append(ptr, ptr)

declare ptr @g_list_remove_link(ptr, ptr)
