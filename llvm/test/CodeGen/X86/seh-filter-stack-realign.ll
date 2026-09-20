; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s

; An SEH filter reaches the locals of its parent through llvm.eh.recoverfp plus
; llvm.localrecover. When the parent dynamically realigns its stack, the
; offsets that llvm.localescape hands out are relative to the realigned base
; pointer. The filter therefore has to redo the realignment
;
; Generated from:
;
;   int zero;
;   void *filt_d, *filt_res;
;   void foo();
;   int check(int arg) {
;     __declspec(align(64)) double d[4];
;     int res = 42;
;     __try {
;       foo();
;     } __except (filt_d = &d, filt_res = &res, zero) {
;     }
;     return res + arg;
;   }

@zero = dso_local global i32 0, align 4
@filt_d = dso_local global ptr null, align 8
@filt_res = dso_local global ptr null, align 8

declare dso_local void @Boom()
declare dso_local i32 @__C_specific_handler(...)

define dso_local i32 @check(i32 %arg) personality ptr @__C_specific_handler {
entry:
  %d = alloca [4 x double], align 64
  %res = alloca i32, align 4
  call void (...) @llvm.localescape(ptr %d, ptr %res)
  store i32 42, ptr %res, align 4
  invoke void @Boom()
          to label %__try.cont unwind label %catch.dispatch

catch.dispatch:
  %cs = catchswitch within none [label %__except.ret] unwind to caller

__except.ret:
  %pad = catchpad within %cs [ptr @filt]
  catchret from %pad to label %__try.cont

__try.cont:
  %r = load i32, ptr %res, align 4
  %add = add i32 %r, %arg
  ret i32 %add
}

define internal i32 @filt(ptr %exception_pointers, ptr %frame_pointer) {
entry:
  %fp = call ptr @llvm.eh.recoverfp(ptr @check, ptr %frame_pointer)
  %d = call ptr @llvm.localrecover(ptr @check, ptr %fp, i32 0)
  %res = call ptr @llvm.localrecover(ptr @check, ptr %fp, i32 1)
  store ptr %d, ptr @filt_d, align 8
  store ptr %res, ptr @filt_res, align 8
  %z = load i32, ptr @zero, align 4
  ret i32 %z
}

declare ptr @llvm.eh.recoverfp(ptr, ptr)
declare ptr @llvm.localrecover(ptr, ptr, i32 immarg)
declare void @llvm.localescape(...)

; CHECK-LABEL: check:
; CHECK:         leaq {{[0-9]+}}(%rsp), %rbp
; CHECK:         .seh_endprologue
; CHECK:         andq $-64, %rsp
; CHECK-NEXT:    movq %rsp, %rbx
; CHECK:         .seh_handlerdata
; CHECK-NEXT:    .Lcheck$parent_frame_offset = 0
; CHECK-NEXT:    .Lcheck$parent_frame_align_mask = -64

; CHECK-LABEL: filt:
; CHECK:         leaq .Lcheck$parent_frame_offset(%rdx), [[BASE:%r[a-z0-9]+]]
; CHECK-NEXT:    andq $-64, [[BASE]]
; CHECK-NEXT:    leaq .Lcheck$frame_escape_0([[BASE]]),
; CHECK-NEXT:    leaq .Lcheck$frame_escape_1([[BASE]]),
