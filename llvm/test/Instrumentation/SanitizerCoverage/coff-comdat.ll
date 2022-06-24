; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -sanitizer-coverage-inline-8bit-counters=1 -sanitizer-coverage-pc-table=1 -S | FileCheck %s

; Make sure we use the right comdat groups for COFF to avoid relocations
; against discarded sections. Internal linkage functions are also different from
; ELF. We don't add a module unique identifier.

; Test based on this source:
; int baz(int);
; static int __attribute__((noinline)) bar(int x) {
;   if (x)
;     return baz(x);
;   return 0;
; }
; int foo(int x) {
;   if (baz(0))
;     x = bar(x);
;   return x;
; }

; Both new comdats should no duplicates on COFF.

; CHECK: $foo = comdat nodeduplicate
; CHECK: $bar = comdat nodeduplicate

; Tables for 'foo' should be in the 'foo' comdat.

; CHECK: @__sancov_gen_{{.*}} = private global [1 x i8] zeroinitializer, section ".SCOV$CM", comdat($foo), align 1

; CHECK: @__sancov_gen_{{.*}} = private constant [2 x i64*]
; CHECK-SAME: [i64* bitcast (i32 (i32)* @foo to i64*), i64* inttoptr (i64 1 to i64*)],
; CHECK-SAME: section ".SCOVP$M", comdat($foo), align 8

; Tables for 'bar' should be in the 'bar' comdat.

; CHECK: @__sancov_gen_{{.*}} = private global [1 x i8] zeroinitializer, section ".SCOV$CM", comdat($bar), align 1

; CHECK: @__sancov_gen_{{.*}} = private constant [2 x i64*]
; CHECK-SAME: [i64* bitcast (i32 (i32)* @bar to i64*), i64* inttoptr (i64 1 to i64*)],
; CHECK-SAME: section ".SCOVP$M", comdat($bar), align 8

; 'foo' and 'bar' should be in their new comdat groups.

; CHECK: define dso_local i32 @foo(i32 %x){{.*}} comdat {
; CHECK: define internal fastcc i32 @bar(i32 %x){{.*}} comdat {

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.14.26433"

; Function Attrs: nounwind uwtable
define dso_local i32 @foo(i32 %x) local_unnamed_addr #0 {
entry:
  %call = tail call i32 @baz(i32 0) #3
  %tobool = icmp eq i32 %call, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %call1 = tail call fastcc i32 @bar(i32 %x)
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %x.addr.0 = phi i32 [ %call1, %if.then ], [ %x, %entry ]
  ret i32 %x.addr.0
}

declare dso_local i32 @baz(i32) local_unnamed_addr #1

; Function Attrs: noinline nounwind uwtable
define internal fastcc i32 @bar(i32 %x) unnamed_addr #2 {
entry:
  %tobool = icmp eq i32 %x, 0
  br i1 %tobool, label %return, label %if.then

if.then:                                          ; preds = %entry
  %call = tail call i32 @baz(i32 %x) #3
  br label %return

return:                                           ; preds = %entry, %if.then
  %retval.0 = phi i32 [ %call, %if.then ], [ 0, %entry ]
  ret i32 %retval.0
}

attributes #0 = { nounwind uwtable }
attributes #1 = { "asdf" }
attributes #2 = { noinline nounwind uwtable }
attributes #3 = { nounwind }
