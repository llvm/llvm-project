; RUN: opt -S -verify-memoryssa -passes=loop-sink < %s | FileCheck %s
; RUN: opt -S -verify-memoryssa -aa-pipeline=basic-aa -passes=loop-sink < %s | FileCheck %s

; CHECK: pr39570
; Make sure not to assert.

%0 = type { i32, ptr, %2, ptr, ptr }
%1 = type { ptr }
%2 = type { ptr }
%3 = type { %4, i32, ptr }
%4 = type { ptr, i32 }
%5 = type opaque
%6 = type { %7, ptr, ptr, i8, %2, ptr }
%7 = type <{ %8, ptr, %10, i32, ptr, ptr, ptr, %27, %28, i16 }>
%8 = type { ptr }
%9 = type opaque
%10 = type { %11, %16, %18, %19 }
%11 = type { ptr, i32, i32, ptr }
%12 = type { ptr }
%13 = type { ptr, ptr }
%14 = type { %15, i32 }
%15 = type { ptr, i32, i32, ptr }
%16 = type { ptr, i32, i32, ptr }
%17 = type { %13, ptr }
%18 = type { ptr, i32, i32, ptr }
%19 = type { %20, %21, ptr, float, i32, i32, %22, %22, %24, i32, i32 }
%20 = type { i8 }
%21 = type { i8 }
%22 = type { ptr, ptr, ptr }
%23 = type opaque
%24 = type { ptr, i32, i32, ptr }
%25 = type { ptr, i32, i32, ptr }
%26 = type opaque
%27 = type { ptr }
%28 = type { %29, i32, i32, ptr }
%29 = type { %30 }
%30 = type { ptr }
%31 = type opaque
%32 = type { ptr }
%33 = type <{ %8, ptr, %10, i32, ptr, ptr, ptr, %27, %28, i16, [2 x i8] }>

define dso_local void @pr39570() local_unnamed_addr align 2 personality ptr @__gxx_personality_v0 !prof !1 {
  br i1 undef, label %8, label %1, !prof !2

; <label>:1:                                      ; preds = %0
  %2 = load ptr, ptr undef, align 4
  br label %3

; <label>:3:                                      ; preds = %7, %1
  %4 = getelementptr inbounds %0, ptr %2, i32 undef, i32 0
  br label %5

; <label>:5:                                      ; preds = %3
  %6 = getelementptr inbounds %0, ptr %2, i32 undef, i32 4
  br i1 undef, label %18, label %7, !prof !3

; <label>:7:                                      ; preds = %5
  br label %3

; <label>:8:                                      ; preds = %0
  invoke void @baz()
          to label %9 unwind label %12

; <label>:9:                                      ; preds = %8
  invoke void @bar()
          to label %17 unwind label %10

; <label>:10:                                     ; preds = %9
  %11 = landingpad { ptr, i32 }
          catch ptr null
  unreachable

; <label>:12:                                     ; preds = %8
  %13 = landingpad { ptr, i32 }
          cleanup
  invoke void @bar()
          to label %16 unwind label %14

; <label>:14:                                     ; preds = %12
  %15 = landingpad { ptr, i32 }
          catch ptr null
  unreachable

; <label>:16:                                     ; preds = %12
  resume { ptr, i32 } %13

; <label>:17:                                     ; preds = %9
  br label %18

; <label>:18:                                     ; preds = %17, %5
  invoke void @baz()
          to label %19 unwind label %20

; <label>:19:                                     ; preds = %18
  invoke void @bar()
          to label %22 unwind label %20

; <label>:20:                                     ; preds = %19
  %21 = landingpad { ptr, i32 }
          catch ptr null
  unreachable

; <label>:22:                                     ; preds = %19
  ret void
}

declare dso_local i32 @__gxx_personality_v0(...)
declare dso_local void @bar() local_unnamed_addr
declare dso_local void @baz() local_unnamed_addr align 2

!1 = !{!"function_entry_count", i64 0}
!2 = !{!"branch_weights", i32 1, i32 3215551}
!3 = !{!"branch_weights", i32 3215551, i32 1}
