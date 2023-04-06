; RUN: opt -S -mtriple=x86_64-unknown-unknown -passes=tlshoist --relocation-model=pic --tls-load-hoist=true -o - %s | FileCheck %s --check-prefix=HOIST0
; RUN: opt -S -mtriple=x86_64-unknown-unknown -passes=tlshoist --relocation-model=pic -o - %s | FileCheck %s --check-prefix=HOIST2

$_ZTW5thl_x = comdat any

@thl_x = thread_local global i32 0, align 4

; Function Attrs: mustprogress uwtable
define i32 @_Z2f1i(i32 %c) local_unnamed_addr #0 {
entry:
  %0 = load i32, ptr @thl_x, align 4
  %call = tail call i32 @_Z5gfunci(i32 %0)
  %1 = load i32, ptr @thl_x, align 4
  %call1 = tail call i32 @_Z5gfunci(i32 %1)
  ret i32 1
}

;HOIST0-LABEL: _Z2f1i
;HOIST0:     entry:
;HOIST0-NEXT:  %tls_bitcast = bitcast ptr @thl_x to ptr
;HOIST0-NEXT:  %0 = load i32, ptr %tls_bitcast, align 4
;HOIST0-NEXT:  %call = tail call i32 @_Z5gfunci(i32 %0)
;HOIST0-NEXT:  %1 = load i32, ptr %tls_bitcast, align 4
;HOIST0-NEXT:  %call1 = tail call i32 @_Z5gfunci(i32 %1)
;HOIST0-NEXT:  ret i32 1

;HOIST2-LABEL: _Z2f1i
;HOIST2:     entry:
;HOIST2-NEXT:  %0 = load i32, ptr @thl_x, align 4
;HOIST2-NEXT:  %call = tail call i32 @_Z5gfunci(i32 %0)
;HOIST2-NEXT:  %1 = load i32, ptr @thl_x, align 4
;HOIST2-NEXT:  %call1 = tail call i32 @_Z5gfunci(i32 %1)
;HOIST2-NEXT:  ret i32 1

declare i32 @_Z5gfunci(i32) local_unnamed_addr #1

; Function Attrs: uwtable
define weak_odr hidden ptr @_ZTW5thl_x() local_unnamed_addr #2 comdat {
  ret ptr @thl_x
}

attributes #0 = { mustprogress uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { uwtable "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 7, !"uwtable", i32 1}
