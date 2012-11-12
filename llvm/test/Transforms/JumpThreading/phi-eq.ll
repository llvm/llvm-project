; RUN: llvm-as < %s | opt -jump-threading | llvm-dis | FileCheck %s
; Test whether two consecutive switches with identical structures assign the
; proper value to the proper variable.  This is really testing 
; Instruction::isIdenticalToWhenDefined, as previously that function was 
; returning true if the value part of the operands of two phis were identical, 
; even if the incoming blocks were not.
; NB: this function should be pruned down more.

%struct._GList = type { i8*, %struct._GList*, %struct._GList* }
%struct.filter_def = type { i8*, i8* }

@capture_filters = external hidden global %struct._GList*, align 8
@display_filters = external hidden global %struct._GList*, align 8
@.str2 = external hidden unnamed_addr constant [10 x i8], align 1
@__PRETTY_FUNCTION__.copy_filter_list = external hidden unnamed_addr constant [62 x i8], align 1
@.str12 = external hidden unnamed_addr constant [22 x i8], align 1
@.str13 = external hidden unnamed_addr constant [31 x i8], align 1
@capture_edited_filters = external hidden global %struct._GList*, align 8
@display_edited_filters = external hidden global %struct._GList*, align 8
@__PRETTY_FUNCTION__.get_filter_list = external hidden unnamed_addr constant [44 x i8], align 1

declare void @g_assertion_message(i8*, i8*, i32, i8*, i8*) noreturn

declare void @g_free(i8*)

declare %struct._GList* @g_list_first(%struct._GList*)

declare noalias i8* @g_malloc(i64)

define void @copy_filter_list(i32 %dest_type, i32 %src_type) nounwind uwtable ssp {
entry:
  br label %do.body

do.body:                                          ; preds = %entry
  %cmp = icmp ne i32 %dest_type, %src_type
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %do.body
  br label %if.end

if.else:                                          ; preds = %do.body
  call void @g_assertion_message_expr(i8* null, i8* getelementptr inbounds ([10 x i8]* @.str2, i32 0, i32 0), i32 581, i8* getelementptr inbounds ([62 x i8]* @__PRETTY_FUNCTION__.copy_filter_list, i32 0, i32 0), i8* getelementptr inbounds ([22 x i8]* @.str12, i32 0, i32 0)) noreturn
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
  call void @g_assertion_message(i8* null, i8* getelementptr inbounds ([10 x i8]* @.str2, i32 0, i32 0), i32 408, i8* getelementptr inbounds ([44 x i8]* @__PRETTY_FUNCTION__.get_filter_list, i32 0, i32 0), i8* null) noreturn nounwind
  unreachable

get_filter_list.exit:                             ; preds = %sw.bb3.i, %sw.bb2.i, %sw.bb1.i, %sw.bb.i
  %0 = phi %struct._GList** [ @display_edited_filters, %sw.bb3.i ], [ @capture_edited_filters, %sw.bb2.i ], [ @display_filters, %sw.bb1.i ], [ @capture_filters, %sw.bb.i ]
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
  call void @g_assertion_message(i8* null, i8* getelementptr inbounds ([10 x i8]* @.str2, i32 0, i32 0), i32 408, i8* getelementptr inbounds ([44 x i8]* @__PRETTY_FUNCTION__.get_filter_list, i32 0, i32 0), i8* null) noreturn nounwind
  unreachable

; CHECK: get_filter_list.exit
get_filter_list.exit6:                            ; preds = %sw.bb3.i4, %sw.bb2.i3, %sw.bb1.i2, %sw.bb.i1
  %1 = phi %struct._GList** [ @display_edited_filters, %sw.bb3.i4 ], [ @capture_edited_filters, %sw.bb2.i3 ], [ @display_filters, %sw.bb1.i2 ], [ @capture_filters, %sw.bb.i1 ]
; CHECK: %2 = load
  %2 = load %struct._GList** %1, align 8
; We should have jump-threading insert an additional load here for the value
; coming out of the first switch, which is picked up by a subsequent phi
; CHECK: {{%\.pr = load %[^%]* %0}}
; CHECK-NEXT:  br label %while.cond
  br label %while.cond

; CHECK: while.cond
while.cond:                                       ; preds = %while.body, %get_filter_list.exit6
; CHECK: {{= phi .*%.pr}}
  %3 = load %struct._GList** %0, align 8
; CHECK: tobool
  %tobool = icmp ne %struct._GList* %3, null
  br i1 %tobool, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %4 = load %struct._GList** %0, align 8
  %5 = load %struct._GList** %0, align 8
  %call2 = call %struct._GList* @g_list_first(%struct._GList* %5)
  %data.i = getelementptr inbounds %struct._GList* %call2, i32 0, i32 0
  %6 = load i8** %data.i, align 8
  %7 = bitcast i8* %6 to %struct.filter_def*
  %name.i = getelementptr inbounds %struct.filter_def* %7, i32 0, i32 0
  %8 = load i8** %name.i, align 8
  call void @g_free(i8* %8) nounwind
  %strval.i = getelementptr inbounds %struct.filter_def* %7, i32 0, i32 1
  %9 = load i8** %strval.i, align 8
  call void @g_free(i8* %9) nounwind
  %10 = bitcast %struct.filter_def* %7 to i8*
  call void @g_free(i8* %10) nounwind
  %call.i = call %struct._GList* @g_list_remove_link(%struct._GList* %4, %struct._GList* %call2) nounwind
  store %struct._GList* %call.i, %struct._GList** %0, align 8
  br label %while.cond

while.end:                                        ; preds = %while.cond
  br label %do.body4

do.body4:                                         ; preds = %while.end
  %11 = load %struct._GList** %0, align 8
  %call5 = call i32 @g_list_length(%struct._GList* %11)
  %cmp6 = icmp eq i32 %call5, 0
  br i1 %cmp6, label %if.then7, label %if.else8

if.then7:                                         ; preds = %do.body4
  br label %if.end9

if.else8:                                         ; preds = %do.body4
  call void @g_assertion_message_expr(i8* null, i8* getelementptr inbounds ([10 x i8]* @.str2, i32 0, i32 0), i32 600, i8* getelementptr inbounds ([62 x i8]* @__PRETTY_FUNCTION__.copy_filter_list, i32 0, i32 0), i8* getelementptr inbounds ([31 x i8]* @.str13, i32 0, i32 0)) noreturn
  unreachable

if.end9:                                          ; preds = %if.then7
  br label %do.end10

do.end10:                                         ; preds = %if.end9
  br label %while.cond11

while.cond11:                                     ; preds = %cond.end, %do.end10
  %cond10 = phi %struct._GList* [ %cond, %cond.end ], [ %2, %do.end10 ]
  %tobool12 = icmp ne %struct._GList* %cond10, null
  br i1 %tobool12, label %while.body13, label %while.end16

while.body13:                                     ; preds = %while.cond11
  %data = getelementptr inbounds %struct._GList* %cond10, i32 0, i32 0
  %12 = load i8** %data, align 8
  %13 = bitcast i8* %12 to %struct.filter_def*
  %14 = load %struct._GList** %0, align 8
  %name = getelementptr inbounds %struct.filter_def* %13, i32 0, i32 0
  %15 = load i8** %name, align 8
  %strval = getelementptr inbounds %struct.filter_def* %13, i32 0, i32 1
  %16 = load i8** %strval, align 8
  %call.i7 = call noalias i8* @g_malloc(i64 16) nounwind
  %17 = bitcast i8* %call.i7 to %struct.filter_def*
  %call1.i = call noalias i8* @g_strdup(i8* %15) nounwind
  %name.i8 = getelementptr inbounds %struct.filter_def* %17, i32 0, i32 0
  store i8* %call1.i, i8** %name.i8, align 8
  %call2.i = call noalias i8* @g_strdup(i8* %16) nounwind
  %strval.i9 = getelementptr inbounds %struct.filter_def* %17, i32 0, i32 1
  store i8* %call2.i, i8** %strval.i9, align 8
  %18 = bitcast %struct.filter_def* %17 to i8*
  %call3.i = call %struct._GList* @g_list_append(%struct._GList* %14, i8* %18) nounwind
  store %struct._GList* %call3.i, %struct._GList** %0, align 8
  %tobool15 = icmp ne %struct._GList* %cond10, null
  br i1 %tobool15, label %cond.true, label %cond.false

cond.true:                                        ; preds = %while.body13
  %next = getelementptr inbounds %struct._GList* %cond10, i32 0, i32 1
  %19 = load %struct._GList** %next, align 8
  br label %cond.end

cond.false:                                       ; preds = %while.body13
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi %struct._GList* [ %19, %cond.true ], [ null, %cond.false ]
  br label %while.cond11

while.end16:                                      ; preds = %while.cond11
  ret void
}

declare void @g_assertion_message_expr(i8*, i8*, i32, i8*, i8*) noreturn

declare i32 @g_list_length(%struct._GList*)

declare noalias i8* @g_strdup(i8*)

declare %struct._GList* @g_list_append(%struct._GList*, i8*)

declare %struct._GList* @g_list_remove_link(%struct._GList*, %struct._GList*)
