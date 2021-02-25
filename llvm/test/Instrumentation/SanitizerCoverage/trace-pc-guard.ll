; RUN: opt < %s -sancov -sanitizer-coverage-level=4 -sanitizer-coverage-trace-pc-guard -mtriple=x86_64 -S -enable-new-pm=0 | FileCheck %s --check-prefixes=CHECK,ELF
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=4 -sanitizer-coverage-trace-pc-guard -mtriple=x86_64 -S | FileCheck %s --check-prefixes=CHECK,ELF

; RUN: opt < %s -sancov -sanitizer-coverage-level=4 -sanitizer-coverage-trace-pc-guard -mtriple=aarch64-apple-darwin -S -enable-new-pm=0 | FileCheck %s --check-prefixes=CHECK,MACHO
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=4 -sanitizer-coverage-trace-pc-guard -mtriple=aarch64-apple-darwin -S | FileCheck %s --check-prefixes=CHECK,MACHO

; ELF:        $foo = comdat any
; ELF:        $CallViaVptr = comdat any
; ELF:        @__sancov_gen_ = private global [3 x i32] zeroinitializer, section "__sancov_guards", comdat($foo), align 4, !associated !0
; ELF-NEXT:   @__sancov_gen_.1 = private global [1 x i32] zeroinitializer, section "__sancov_guards", comdat($CallViaVptr), align 4, !associated !1

; MACHO:      @__sancov_gen_ = private global [3 x i32] zeroinitializer, section "__DATA,__sancov_guards", align 4, !associated !0
; MACHO-NEXT: @__sancov_gen_.1 = private global [1 x i32] zeroinitializer, section "__DATA,__sancov_guards", align 4, !associated !1

; ELF-NOT:    @llvm.used =
; MACHO:      @llvm.used = appending global [2 x i8*] [i8* bitcast ([3 x i32]* @__sancov_gen_ to i8*), i8* bitcast ([1 x i32]* @__sancov_gen_.1 to i8*)], section "llvm.metadata"
; CHECK:      @llvm.compiler.used = appending global [2 x i8*] [i8* bitcast ([3 x i32]* @__sancov_gen_ to i8*), i8* bitcast ([1 x i32]* @__sancov_gen_.1 to i8*)], section "llvm.metadata"

; CHECK-LABEL: define void @foo
; CHECK:         call void @__sanitizer_cov_trace_pc
; CHECK:         ret void

define void @foo(i32* %a) sanitize_address {
entry:
  %tobool = icmp eq i32* %a, null
  br i1 %tobool, label %if.end, label %if.then

  if.then:                                          ; preds = %entry
  store i32 0, i32* %a, align 4
  br label %if.end

  if.end:                                           ; preds = %entry, %if.then
  ret void
}

; CHECK-LABEL: define void @CallViaVptr
; CHECK:         call void @__sanitizer_cov_trace_pc_indir
; CHECK:         call void @__sanitizer_cov_trace_pc_indir
; CHECK:         ret void

%struct.StructWithVptr = type { i32 (...)** }

define void @CallViaVptr(%struct.StructWithVptr* %foo) uwtable sanitize_address {
entry:
  %0 = bitcast %struct.StructWithVptr* %foo to void (%struct.StructWithVptr*)***
  %vtable = load void (%struct.StructWithVptr*)**, void (%struct.StructWithVptr*)*** %0, align 8
  %1 = load void (%struct.StructWithVptr*)*, void (%struct.StructWithVptr*)** %vtable, align 8
  tail call void %1(%struct.StructWithVptr* %foo)
  tail call void %1(%struct.StructWithVptr* %foo)
  tail call void asm sideeffect "", ""()
  ret void
}

; ELF-LABEL: define internal void @sancov.module_ctor_trace_pc_guard() comdat {
; MACHO-LABEL: define internal void @sancov.module_ctor_trace_pc_guard() {
