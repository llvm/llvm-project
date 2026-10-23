; RUN: llvm-reduce %s -o %t --abort-on-invalid-reduction \
; RUN:   --delta-passes=operands-to-args \
; RUN:   --test FileCheck --test-arg %s --test-arg --check-prefix=INTERESTING \
; RUN:   --test-arg --input-file
; RUN: FileCheck %s --input-file %t --check-prefix=REDUCED

; INTERESTING: landingpad
; REDUCED-LABEL: define void @f()
; REDUCED:       landingpad { ptr, i32 }
; REDUCED-NEXT:    catch ptr @typeinfo

@typeinfo = external constant ptr

declare void @g()
declare i32 @personality(...)

define void @f() personality ptr @personality {
entry:
  invoke void @g()
          to label %return unwind label %lpad

lpad:
  %result = landingpad { ptr, i32 }
          catch ptr @typeinfo
  unreachable

return:
  ret void
}
