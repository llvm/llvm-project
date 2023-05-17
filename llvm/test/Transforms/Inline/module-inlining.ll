; modify_value will be inlined into main. With just the inliner pass, at most 
; some trivial DCE would happen, which in this case doesn't modify post-inlined
; main much.
; In contrast, with the full set of module inliner-related passes, at the end of
; inlining (incl. function cleanups ran after inlining), main will be reduced to
; a 'ret 10'
;
; RUN: opt -passes=inline -S < %s | FileCheck %s --check-prefix=INLINE --check-prefix=CHECK
; RUN: opt -passes=inliner-wrapper -S < %s | FileCheck %s --check-prefix=INLINE --check-prefix=CHECK
; RUN: opt -passes=scc-oz-module-inliner -S < %s | FileCheck %s --check-prefix=MODULE --check-prefix=CHECK

define void @modify_value(ptr %v) {
    %f = getelementptr { i32, float }, ptr %v, i64 0, i32 0
    store i32 10, ptr %f
    ret void
}

define i32 @main() {
    %my_val = alloca {i32, float}
    call void @modify_value(ptr %my_val)
    %f = getelementptr { i32, float }, ptr %my_val, i64 0, i32 0
    %ret = load i32, ptr %f
    ret i32 %ret
}

; CHECK-LABEL: @main
; INLINE-NEXT: %my_val = alloca
; MODULE-NEXT: ret i32 10