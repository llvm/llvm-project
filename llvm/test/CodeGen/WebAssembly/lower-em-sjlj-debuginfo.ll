; RUN: opt < %s -wasm-lower-em-ehsjlj -enable-emscripten-sjlj -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

%struct.__jmp_buf_tag = type { [6 x i32], i32, [32 x i32] }

; Basic debug info test. All existing instructions have debug info and inserted
; 'malloc' and 'free' calls take debug info from the next instruction.
define void @setjmp_debug_info0() !dbg !3 {
; CHECK-LABEL: @setjmp_debug_info0
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16, !dbg !4
  %arraydecay = getelementptr inbounds [1 x %struct.__jmp_buf_tag], ptr %buf, i32 0, i32 0, !dbg !5
  %call = call i32 @setjmp(ptr %arraydecay) #0, !dbg !6
  call void @foo(), !dbg !7
  ret void, !dbg !8
; CHECK: entry:
  ; CHECK-NEXT: %functionInvocationId = alloca i32, align 4, !dbg ![[DL0:.*]]

; CHECK: entry.split:
  ; CHECK: alloca {{.*}}, !dbg ![[DL0]]
  ; CHECK: call void @__wasm_setjmp{{.*}}, !dbg ![[DL1:.*]]
  ; CHECK-NEXT: br {{.*}}, !dbg ![[DL2:.*]]

; CHECK: entry.split.split:
  ; CHECK: call {{.*}} void @__invoke_void{{.*}}, !dbg ![[DL2]]

; CHECK: entry.split.split.split:

; CHECK: if.then1:
  ; CHECK: call i32 @__wasm_setjmp_test{{.*}}, !dbg ![[DL2]]

; CHECK: if.end:

; CHECK: call.em.longjmp:
  ; CHECK: call void @emscripten_longjmp{{.*}}, !dbg ![[DL2]]

; CHECK: if.end2:
  ; CHECK: call void @setTempRet0{{.*}}, !dbg ![[DL2]]
}

declare void @foo()
; Function Attrs: returns_twice
declare i32 @setjmp(ptr) #0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !DIFile(filename: "lower-em-sjlj.c", directory: "test")
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1)
!3 = distinct !DISubprogram(name: "setjmp_debug_info0", unit:!2, file: !1, line: 1)
!4 = !DILocation(line:2, scope: !3)
!5 = !DILocation(line:3, scope: !3)
!6 = !DILocation(line:4, scope: !3)
!7 = !DILocation(line:5, scope: !3)
!8 = !DILocation(line:6, scope: !3)
