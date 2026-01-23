; RUN: llc %s --mtriple=aarch64-pc-windows-msvc -o - | FileCheck %s

; Regression test for handling MSVC C++ exceptions when there's an aligned
; object on the stack.

; Generated from this C++ code:
; https://godbolt.org/z/cGzGfqq34
; > clang --target=aarch64-pc-windows-msvc test.cpp
; ```
; // Large object: alignment seems to be important?
; struct alignas(128) BigObj {
;     int value;
;     // Destructor so it's kept alive.
;     ~BigObj() { }
; };
;
; // Exception type need to be large enough to not fit in a register.
; struct Error {
;     int value;
;     int padding[3];
; };
;
; int main() {
;     BigObj bo{};
;
;     try {
;         throw Error { 42, {0, 0, 0} };
;     } catch (const Error& e) {
;         return e.value;
;     }
;     return 0;
; }
; ```

; CHECK-LABEL:  main:
; CHECK:        sub x[[SPTMP:[0-9]+]], sp, #336
; CHECK:        and sp, x[[SPTMP]], #0xffffffffffffff80
; CHECK:        mov x[[FP:[0-9]+]], sp
; CHECK:        str wzr, [x[[FP]], #332]

; CHECK-LABEL:  "?catch$3@?0?main@4HA":
; CHECK:        str	w8, [x[[FP]], #332]
; CHECK-NEXT:   .seh_startepilogue
; CHECK:        ret

; CHECK-LABEL:  $cppxdata$main:
; CHECK:        .word   -16                             // UnwindHelp
; CHECK-LABEL:  $handlerMap$0$main:
; CHECK-NEXT:   .word   8                               // Adjectives
; CHECK-NEXT:   .word   "??_R0?AUError@@@8"@IMGREL      // Type
; CHECK-NEXT:   .word   -8                              // CatchObjOffset
; CHECK-NEXT:   .word   "?catch$3@?0?main@4HA"@IMGREL   // Handler

%rtti.TypeDescriptor11 = type { ptr, ptr, [12 x i8] }
%eh.CatchableType = type { i32, i32, i32, i32, i32, i32, i32 }
%eh.CatchableTypeArray.1 = type { i32, [1 x i32] }
%eh.ThrowInfo = type { i32, i32, i32, i32 }
%struct.BigObj = type { i32, [124 x i8] }
%struct.Error = type { i32, [3 x i32] }

$"??1BigObj@@QEAA@XZ" = comdat any

$"??_R0?AUError@@@8" = comdat any

$"_CT??_R0?AUError@@@816" = comdat any

$"_CTA1?AUError@@" = comdat any

$"_TI1?AUError@@" = comdat any

@"??_7type_info@@6B@" = external constant ptr
@"??_R0?AUError@@@8" = linkonce_odr global %rtti.TypeDescriptor11 { ptr @"??_7type_info@@6B@", ptr null, [12 x i8] c".?AUError@@\00" }, comdat
@__ImageBase = external dso_local constant i8
@"_CT??_R0?AUError@@@816" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"??_R0?AUError@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 16, i32 0 }, section ".xdata", comdat
@"_CTA1?AUError@@" = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.1 { i32 1, [1 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CT??_R0?AUError@@@816" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32)] }, section ".xdata", comdat
@"_TI1?AUError@@" = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CTA1?AUError@@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, section ".xdata", comdat

define dso_local noundef i32 @main() personality ptr @__CxxFrameHandler3 {
entry:
  %retval = alloca i32, align 4
  %bo = alloca %struct.BigObj, align 128
  %tmp = alloca %struct.Error, align 4
  %e = alloca ptr, align 8
  %cleanup.dest.slot = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  call void @llvm.memset.p0.i64(ptr align 128 %bo, i8 0, i64 128, i1 false)
  %value = getelementptr inbounds nuw %struct.BigObj, ptr %bo, i32 0, i32 0
  %value1 = getelementptr inbounds nuw %struct.Error, ptr %tmp, i32 0, i32 0
  store i32 42, ptr %value1, align 4
  %padding = getelementptr inbounds nuw %struct.Error, ptr %tmp, i32 0, i32 1
  store i32 0, ptr %padding, align 4
  %arrayinit.element = getelementptr inbounds i32, ptr %padding, i64 1
  store i32 0, ptr %arrayinit.element, align 4
  %arrayinit.element2 = getelementptr inbounds i32, ptr %padding, i64 2
  store i32 0, ptr %arrayinit.element2, align 4
  invoke void @_CxxThrowException(ptr %tmp, ptr @"_TI1?AUError@@") #3
          to label %unreachable unwind label %catch.dispatch

catch.dispatch:
  %0 = catchswitch within none [label %catch] unwind label %ehcleanup

catch:
  %1 = catchpad within %0 [ptr @"??_R0?AUError@@@8", i32 8, ptr %e]
  %2 = load ptr, ptr %e, align 8
  %value3 = getelementptr inbounds nuw %struct.Error, ptr %2, i32 0, i32 0
  %3 = load i32, ptr %value3, align 4
  store i32 %3, ptr %retval, align 4
  store i32 1, ptr %cleanup.dest.slot, align 4
  catchret from %1 to label %catchret.dest

catchret.dest:
  br label %cleanup

try.cont:
  store i32 0, ptr %retval, align 4
  store i32 1, ptr %cleanup.dest.slot, align 4
  br label %cleanup

cleanup:
  call void @"??1BigObj@@QEAA@XZ"(ptr noundef nonnull align 128 dereferenceable(4) %bo) #4
  %4 = load i32, ptr %retval, align 4
  ret i32 %4

ehcleanup:
  %5 = cleanuppad within none []
  call void @"??1BigObj@@QEAA@XZ"(ptr noundef nonnull align 128 dereferenceable(4) %bo) [ "funclet"(token %5) ]
  cleanupret from %5 unwind to caller

unreachable:
  unreachable
}

declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #1

declare dso_local void @_CxxThrowException(ptr, ptr)

declare dso_local i32 @__CxxFrameHandler3(...)

define linkonce_odr dso_local void @"??1BigObj@@QEAA@XZ"(ptr noundef nonnull align 128 dereferenceable(4) %this) unnamed_addr comdat {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  ret void
}
