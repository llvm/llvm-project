; RUN: opt -hotcoldsplit -hotcoldsplit-threshold=2 -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; Check that these functions are not split. Outlined functions are called from a
; basic block named codeRepl.

; The cold region is too small to split.
; CHECK-LABEL: @foo
; CHECK-NOT: foo.cold.1
define void @foo() {
entry:
  br i1 undef, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  unreachable

if.end:                                           ; preds = %entry
  ret void
}

; The cold region is still too small to split.
; CHECK-LABEL: @bar
; CHECK-NOT: bar.cold.1
define void @bar() {
entry:
  br i1 undef, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @sink()
  ret void

if.end:                                           ; preds = %entry
  ret void
}

; Make sure we don't try to outline the entire function.
; CHECK-LABEL: @fun
; CHECK-NOT: fun.cold.1
define void @fun() {
entry:
  br i1 undef, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @sink()
  br label %if.end

if.end:                                           ; preds = %entry
  ret void
}

; Do not split `noinline` functions.
; CHECK-LABEL: @noinline_func
; CHECK-NOT: noinline_func.cold.1
define void @noinline_func() noinline {
entry:
  br i1 undef, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @sink()
  br label %if.end

if.end:                                           ; preds = %entry
  ret void
}

; Do not split `alwaysinline` functions.
; CHECK-LABEL: @alwaysinline_func
; CHECK-NOT: alwaysinline_func.cold.1
define void @alwaysinline_func() alwaysinline {
entry:
  br i1 undef, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @sink()
  br label %if.end

if.end:                                           ; preds = %entry
  ret void
}

; Don't outline infinite loops.
; CHECK-LABEL: @infinite_loop
; CHECK-NOT: infinite_loop.cold.1
define void @infinite_loop() {
entry:
  br label %loop

loop:
  call void @sink()
  br label %loop
}

; Don't count debug intrinsics towards the outlining threshold.
; CHECK-LABEL: @dont_count_debug_intrinsics
; CHECK-NOT: dont_count_debug_intrinsics.cold.1
define void @dont_count_debug_intrinsics(i32 %arg1) !dbg !6 {
entry:
  %var = add i32 0, 0, !dbg !11
  br i1 undef, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  ret void

if.end:                                           ; preds = %entry
  call void @llvm.dbg.value(metadata i32 %arg1, metadata !9, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.value(metadata i32 %arg1, metadata !9, metadata !DIExpression()), !dbg !11
  call void @sink()
  ret void
}

; CHECK-LABEL: @sanitize_address
; CHECK-NOT: sanitize_address.cold.1
define void @sanitize_address() sanitize_address {
entry:
  br i1 undef, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @sink()
  ret void

if.end:                                           ; preds = %entry
  ret void
}

; CHECK-LABEL: @sanitize_hwaddress
; CHECK-NOT: sanitize_hwaddress.cold.1
define void @sanitize_hwaddress() sanitize_hwaddress {
entry:
  br i1 undef, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @sink()
  ret void

if.end:                                           ; preds = %entry
  ret void
}

; CHECK-LABEL: @sanitize_thread
; CHECK-NOT: sanitize_thread.cold.1
define void @sanitize_thread() sanitize_thread {
entry:
  br i1 undef, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @sink()
  ret void

if.end:                                           ; preds = %entry
  ret void
}

; CHECK-LABEL: @sanitize_memory
; CHECK-NOT: sanitize_memory.cold.1
define void @sanitize_memory() sanitize_memory {
entry:
  br i1 undef, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @sink()
  ret void

if.end:                                           ; preds = %entry
  ret void
}

declare void @llvm.trap() cold noreturn

; CHECK-LABEL: @nosanitize_call
; CHECK-NOT: nosanitize_call.cold.1
define void @nosanitize_call() sanitize_memory {
entry:
  br i1 undef, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @llvm.trap(), !nosanitize !2
  unreachable

if.end:                                           ; preds = %entry
  ret void
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

declare void @sink() cold

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!3, !4}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "<stdin>", directory: "/")
!2 = !{}
!3 = !{i32 7}
!4 = !{i32 1}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "dont_count_debug_intrinsics", linkageName: "dont_count_debug_intrinsics", scope: null, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !0, retainedNodes: !8)
!7 = !DISubroutineType(types: !2)
!8 = !{!9}
!9 = !DILocalVariable(name: "1", scope: !6, file: !1, line: 1, type: !10)
!10 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!11 = !DILocation(line: 1, column: 1, scope: !6)
