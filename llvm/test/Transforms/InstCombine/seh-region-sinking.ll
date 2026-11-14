; RUN: opt -passes=instcombine -verify-each -S < %s | FileCheck %s
; RUN: opt -mtriple=x86_64-pc-windows-msvc -passes='require<profile-summary>,function(codegenprepare)' -verify-each -S < %s | FileCheck %s --check-prefix=CGP
; RUN: sed '/!llvm.module.flags/d' %s | opt -passes=instcombine -verify-each -S | FileCheck %s --check-prefix=NOASYNC

declare void @llvm.seh.try.begin()
declare void @llvm.seh.try.end()
declare i32 @__C_specific_handler(...)

define i32 @no_sink_into_try(ptr %address, i32 %number) personality ptr @__C_specific_handler {
entry:
  %multiply = mul i32 %number, 3
  %add = add i32 %multiply, 7
  invoke void @llvm.seh.try.begin() to label %body unwind label %dispatch
body:
  %value = load volatile i32, ptr %address
  %sum = add i32 %add, %value
  invoke void @llvm.seh.try.end() to label %join unwind label %dispatch
dispatch:
  %switch = catchswitch within none [label %handler] unwind to caller
handler:
  %pad = catchpad within %switch [ptr null]
  catchret from %pad to label %join
join:
  %result = phi i32 [ %sum, %body ], [ -1, %handler ]
  ret i32 %result
}

; CHECK-LABEL: define i32 @no_sink_into_try(
; CHECK: entry:
; CHECK-NEXT: [[MULTIPLY:%.*]] = mul i32 %number, 3
; CHECK-NEXT: [[ADD:%.*]] = add i32 [[MULTIPLY]], 7
; CHECK-NEXT: invoke void @llvm.seh.try.begin()

; NOASYNC-LABEL: define i32 @no_sink_into_try(
; NOASYNC: entry:
; NOASYNC-NEXT: invoke void @llvm.seh.try.begin()
; NOASYNC: body:
; NOASYNC-NEXT: [[UNGUARDED_MUL:%.*]] = mul i32 %number, 3
; NOASYNC-NEXT: [[UNGUARDED_ADD:%.*]] = add i32 [[UNGUARDED_MUL]], 7

define i32 @sink_within_try(ptr %address, i32 %number, i1 %condition) personality ptr @__C_specific_handler {
entry:
  invoke void @llvm.seh.try.begin() to label %body unwind label %dispatch
body:
  %multiply = mul i32 %number, 3
  br i1 %condition, label %use, label %end
use:
  %value = load volatile i32, ptr %address
  %sum = add i32 %multiply, %value
  br label %end
end:
  %value.end = phi i32 [ %sum, %use ], [ 0, %body ]
  invoke void @llvm.seh.try.end() to label %join unwind label %dispatch
dispatch:
  %switch = catchswitch within none [label %handler] unwind to caller
handler:
  %pad = catchpad within %switch [ptr null]
  catchret from %pad to label %join
join:
  %result = phi i32 [ %value.end, %end ], [ -1, %handler ]
  ret i32 %result
}

; CHECK-LABEL: define i32 @sink_within_try(
; CHECK: use:
; CHECK-NEXT: [[SAME:%.*]] = mul i32 %number, 3

define i32 @no_sink_out_of_try(i32 %number) personality ptr @__C_specific_handler {
entry:
  invoke void @llvm.seh.try.begin() to label %body unwind label %dispatch
body:
  %multiply = mul i32 %number, 5
  invoke void @llvm.seh.try.end() to label %join unwind label %dispatch
dispatch:
  %switch = catchswitch within none [label %handler] unwind to caller
handler:
  %pad = catchpad within %switch [ptr null]
  catchret from %pad to label %fail
join:
  %result = add i32 %multiply, 7
  ret i32 %result
fail:
  ret i32 -1
}

; CHECK-LABEL: define i32 @no_sink_out_of_try(
; CHECK: body:
; CHECK-NEXT: %multiply = mul i32 %number, 5
; CHECK-NEXT: invoke void @llvm.seh.try.end()

define void @no_sink_cast_or_compare(ptr %address, i64 %wide, i32 %number) personality ptr @__C_specific_handler {
entry:
  %truncate = trunc i64 %wide to i32
  %compare = icmp eq i32 %number, 0
  invoke void @llvm.seh.try.begin() to label %body unwind label %dispatch
body:
  store volatile i32 %truncate, ptr %address
  %select = select i1 %compare, i32 3, i32 4
  store volatile i32 %select, ptr %address
  invoke void @llvm.seh.try.end() to label %exit unwind label %dispatch
dispatch:
  %switch = catchswitch within none [label %handler] unwind to caller
handler:
  %pad = catchpad within %switch [ptr null]
  catchret from %pad to label %exit
exit:
  ret void
}

; CGP-LABEL: define void @no_sink_cast_or_compare(
; CGP: entry:
; CGP-NEXT: %truncate = trunc i64 %wide to i32
; CGP-NEXT: %compare = icmp eq i32 %number, 0
; CGP-NEXT: invoke void @llvm.seh.try.begin()

define void @no_sink_mask(ptr %address, i32 %number) personality ptr @__C_specific_handler {
entry:
  %mask = and i32 %number, 7
  invoke void @llvm.seh.try.begin() to label %body unwind label %dispatch
body:
  %compare = icmp eq i32 %mask, 0
  %select = select i1 %compare, i32 3, i32 4
  store volatile i32 %select, ptr %address
  invoke void @llvm.seh.try.end() to label %exit unwind label %dispatch
dispatch:
  %switch = catchswitch within none [label %handler] unwind to caller
handler:
  %pad = catchpad within %switch [ptr null]
  catchret from %pad to label %exit
exit:
  ret void
}

; CGP-LABEL: define void @no_sink_mask(
; CGP: entry:
; CGP-NEXT: %mask = and i32 %number, 7
; CGP-NEXT: invoke void @llvm.seh.try.begin()

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"eh-asynch", i32 1}