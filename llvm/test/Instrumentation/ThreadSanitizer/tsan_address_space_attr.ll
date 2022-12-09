; RUN: opt < %s -passes=tsan -S | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

; Checks that we do not instrument loads and stores comming from custom address space.
; These result in crashing the compiler.
; int foo(int argc, const char * argv[]) {
;   ptr__attribute__((address_space(256))) *gs_base = (((ptr __attribute__((address_space(256))) *)0));
;   ptr somevalue = gs_base[-1];
;   return somevalue;
; }

define i32 @foo(i32 %argc, ptr %argv) sanitize_thread {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca ptr, align 8
  %gs_base = alloca ptr addrspace(256), align 8
  %somevalue = alloca ptr, align 8
  store i32 0, ptr %retval, align 4
  store i32 %argc, ptr %argc.addr, align 4
  store ptr %argv, ptr %argv.addr, align 8
  store ptr addrspace(256) null, ptr %gs_base, align 8
  %0 = load ptr addrspace(256), ptr %gs_base, align 8
  %arrayidx = getelementptr inbounds ptr, ptr addrspace(256) %0, i64 -1
  %1 = load ptr, ptr addrspace(256) %arrayidx, align 8
  store ptr %1, ptr %somevalue, align 8
  %2 = load ptr, ptr %somevalue, align 8
  %3 = ptrtoint ptr %2 to i32
  ret i32 %3
}
; CHECK-NOT: call void @__tsan_read
; CHECK-NOT: addrspacecast
