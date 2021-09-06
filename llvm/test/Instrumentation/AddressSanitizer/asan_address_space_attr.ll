; RUN: opt < %s -passes='asan-function-pipeline' -S | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

; Checks that we do not instrument loads and stores comming from custom address space.
; These result in invalid (false positive) reports.
; int foo(int argc, const char * argv[]) {
;   void *__attribute__((address_space(256))) *gs_base = (((void * __attribute__((address_space(256))) *)0));
;   void *somevalue = gs_base[-1];
;   return somevalue;
; }

define i32 @foo(i32 %argc, i8** %argv) sanitize_address {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %gs_base = alloca i8* addrspace(256)*, align 8
  %somevalue = alloca i8*, align 8
  store i32 0, i32* %retval, align 4
  store i32 %argc, i32* %argc.addr, align 4
  store i8** %argv, i8*** %argv.addr, align 8
  store i8* addrspace(256)* null, i8* addrspace(256)** %gs_base, align 8
  %0 = load i8* addrspace(256)*, i8* addrspace(256)** %gs_base, align 8
  %arrayidx = getelementptr inbounds i8*, i8* addrspace(256)* %0, i64 -1
  %1 = load i8*, i8* addrspace(256)* %arrayidx, align 8
  store i8* %1, i8** %somevalue, align 8
  %2 = load i8*, i8** %somevalue, align 8
  %3 = ptrtoint i8* %2 to i32
  ret i32 %3
}
; CHECK-NOT: call void @__asan_report_load8
