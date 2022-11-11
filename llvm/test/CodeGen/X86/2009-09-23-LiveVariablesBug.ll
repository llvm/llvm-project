; RUN: llc < %s -mtriple=x86_64-apple-darwin10

; rdar://7247745

%struct._lck_mtx_ = type { %union.anon }
%struct._lck_rw_t_internal_ = type <{ i16, i8, i8, i32, i32, i32 }>
%struct.anon = type { i64, i64, [2 x i8], i8, i8, i32 }
%struct.memory_object = type { i32, i32, ptr }
%struct.memory_object_control = type { i32, i32, ptr }
%struct.memory_object_pager_ops = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct.queue_entry = type { ptr, ptr }
%struct.upl = type { %struct._lck_mtx_, i32, i32, ptr, i64, i32, i64, ptr, i32, ptr }
%struct.upl_page_info = type <{ i32, i8, [3 x i8] }>
%struct.vm_object = type { %struct.queue_entry, %struct._lck_rw_t_internal_, i64, ptr, i32, i32, i32, i32, ptr, ptr, i64, ptr, i64, ptr, i32, i16, i16, [2 x i8], i8, i8, %struct.queue_entry, %struct.queue_entry, i64, i32, i32, i32, ptr, i64, i8, i8, [2 x i8], %struct.queue_entry }
%struct.vm_page = type { %struct.queue_entry, %struct.queue_entry, ptr, ptr, i64, [2 x i8], i8, i8, i32, i8, i8, i8, i8, i32 }
%union.anon = type { %struct.anon }

declare i64 @OSAddAtomic64(i64, ptr) noredzone noimplicitfloat

define i32 @upl_commit_range(ptr %upl, i32 %offset, i32 %size, i32 %flags, ptr %page_list, i32 %count, ptr nocapture %empty) nounwind noredzone noimplicitfloat {
entry:
  br i1 undef, label %if.then, label %if.end

if.end:                                           ; preds = %entry
  br i1 undef, label %if.end143, label %if.then136

if.then136:                                       ; preds = %if.end
  unreachable

if.end143:                                        ; preds = %if.end
  br i1 undef, label %if.else155, label %if.then153

if.then153:                                       ; preds = %if.end143
  br label %while.cond

if.else155:                                       ; preds = %if.end143
  unreachable

while.cond:                                       ; preds = %if.end1039, %if.then153
  br i1 undef, label %if.then1138, label %while.body

while.body:                                       ; preds = %while.cond
  br i1 undef, label %if.end260, label %if.then217

if.then217:                                       ; preds = %while.body
  br i1 undef, label %if.end260, label %if.then230

if.then230:                                       ; preds = %if.then217
  br i1 undef, label %if.then246, label %if.end260

if.then246:                                       ; preds = %if.then230
  br label %if.end260

if.end260:                                        ; preds = %if.then246, %if.then230, %if.then217, %while.body
  br i1 undef, label %if.end296, label %if.then266

if.then266:                                       ; preds = %if.end260
  unreachable

if.end296:                                        ; preds = %if.end260
  br i1 undef, label %if.end1039, label %if.end306

if.end306:                                        ; preds = %if.end296
  br i1 undef, label %if.end796, label %if.then616

if.then616:                                       ; preds = %if.end306
  br i1 undef, label %commit_next_page, label %do.body716

do.body716:                                       ; preds = %if.then616
  %call721 = call i64 @OSAddAtomic64(i64 1, ptr undef) nounwind noredzone noimplicitfloat ; <i64> [#uses=0]
  call void asm sideeffect "movq\090x0($0),%rdi\0A\09movq\090x8($0),%rsi\0A\09.section __DATA, __data\0A\09.globl __dtrace_probeDOLLAR${:uid}4794___vminfo____pgrec\0A\09__dtrace_probeDOLLAR${:uid}4794___vminfo____pgrec:.quad 1f\0A\09.text\0A\091:nop\0A\09nop\0A\09nop\0A\09", "r,~{memory},~{di},~{si},~{dirflag},~{fpsr},~{flags}"(ptr undef) nounwind
  br label %commit_next_page

if.end796:                                        ; preds = %if.end306
  unreachable

commit_next_page:                                 ; preds = %do.body716, %if.then616
  br i1 undef, label %if.end1039, label %if.then1034

if.then1034:                                      ; preds = %commit_next_page
  br label %if.end1039

if.end1039:                                       ; preds = %if.then1034, %commit_next_page, %if.end296
  br label %while.cond

if.then1138:                                      ; preds = %while.cond
  unreachable

if.then:                                          ; preds = %entry
  ret i32 4
}
