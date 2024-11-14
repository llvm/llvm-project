; RUN: opt < %s -passes=rtsan -S | FileCheck %s

define i32 @example(i32 %x) #0 {
entry:
    %retval = alloca i32
    %cmp = icmp sgt i32 %x, 10
    br i1 %cmp, label %then, label %else

then:
    ret i32 1

else:
    ret i32 0
}

attributes #0 = { mustprogress noinline sanitize_realtime optnone ssp uwtable(sync) }

; RealtimeSanitizer pass should insert __rtsan_realtime_enter right after function definition
; CHECK-LABEL: @example(
; CHECK-NEXT: entry:
; CHECK-NEXT: call{{.*}}@__rtsan_realtime_enter

; RealtimeSanitizer pass should insert the call at both function returns
; CHECK-LABEL: then:
; CHECK-NEXT: call{{.*}}@__rtsan_realtime_exit
; CHECK-NEXT: ret i32 1

; CHECK-LABEL: else:
; CHECK-NEXT: call{{.*}}@__rtsan_realtime_exit
; CHECK-NEXT: ret i32 0
