; RUN: llc %s --mtriple=x86_64-pc-windows-msvc -o - | FileCheck %s

; Tests the fixed object layouts when two catchpads re-use the same stack
; allocation for this catch objects.

; Generated from this C++ code, with modifications to the IR (see comments in
; IR):
; https://godbolt.org/z/9qv5Yn68j
; > clang --target=x86_64-pc-windows-msvc test.cpp
; ```
; extern "C" void boom();
; extern "C" int calls_boom();
; {
;     try { boom(); }
;     catch (int& i) { return i; }
;     catch (long& l) { return l; }
;     return 0;
; }
; ```

; Minimum stack alloc is 64 bytes, so no change there.
; CHECK-LABEL:  calls_boom:
; CHECK:        subq    $64, %rsp
; CHECK:        .seh_stackalloc 64

; Both the catch blocks load from the same address.
; CHECK-LABEL:  "?catch$3@?0?calls_boom@4HA":
; CHECK:        movq    -8(%rbp), %rax
; CHECK-LABEL:  "?catch$4@?0?calls_boom@4HA":
; CHECK:        movq    -8(%rbp), %rax

; There's enough space for the UnwindHelp to be at 48 instead of 40
; CHECK-LABEL:  $cppxdata$calls_boom:
; CHECK:        .long   48                              # UnwindHelp

; Both catches have the same object offset.
; CHECK-LABEL:  $handlerMap$0$calls_boom:
; CHECK:        .long   56                              # CatchObjOffset
; CHECK-NEXT:   .long   "?catch$3@?0?calls_boom@4HA"@IMGREL # Handler
; CHECK:        .long   56                              # CatchObjOffset
; CHECK-NEXT:   .long   "?catch$4@?0?calls_boom@4HA"@IMGREL # Handler

%rtti.TypeDescriptor2 = type { ptr, ptr, [3 x i8] }

$"??_R0H@8" = comdat any

$"??_R0J@8" = comdat any

@"??_7type_info@@6B@" = external constant ptr
@"??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { ptr @"??_7type_info@@6B@", ptr null, [3 x i8] c".H\00" }, comdat
@"??_R0J@8" = linkonce_odr global %rtti.TypeDescriptor2 { ptr @"??_7type_info@@6B@", ptr null, [3 x i8] c".J\00" }, comdat

define dso_local i32 @calls_boom() personality ptr @__CxxFrameHandler3 {
entry:
  %retval = alloca i32, align 4
; MODIFICATION: Remove unusued alloca
;  %l = alloca ptr, align 8
  %i = alloca ptr, align 8
  invoke void @boom()
          to label %invoke.cont unwind label %catch.dispatch

catch.dispatch:
  %0 = catchswitch within none [label %catch1, label %catch] unwind to caller

catch1:
  %1 = catchpad within %0 [ptr @"??_R0H@8", i32 8, ptr %i]
  %2 = load ptr, ptr %i, align 8
  %3 = load i32, ptr %2, align 4
  store i32 %3, ptr %retval, align 4
  catchret from %1 to label %catchret.dest2

catch:
; MODIFICATION: Use %i instead of %l
  %4 = catchpad within %0 [ptr @"??_R0J@8", i32 8, ptr %i]
  %5 = load ptr, ptr %i, align 8
  %6 = load i32, ptr %5, align 4
  store i32 %6, ptr %retval, align 4
  catchret from %4 to label %catchret.dest

invoke.cont:
  br label %try.cont

catchret.dest:
  br label %return

catchret.dest2:
  br label %return

try.cont:
  store i32 0, ptr %retval, align 4
  br label %return

return:
  %7 = load i32, ptr %retval, align 4
  ret i32 %7
}

declare dso_local void @boom() #1

declare dso_local i32 @__CxxFrameHandler3(...)
