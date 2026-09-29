; RUN: opt -S -passes='simplifycfg<sink-common-insts>' < %s | FileCheck %s

; The custom-ABI attributes are string attributes, so AttributeSet::intersectWith
; requires them to match exactly before two calls can be sunk into a common
; successor. Sinking calls with different register-preservation contracts would
; silently drop one of them.

; Differing values: both calls must survive in their own blocks.
; CHECK-LABEL: @differing_values(
; CHECK: if:
; CHECK-NEXT: call void %fp() #[[#]]
; CHECK: else:
; CHECK-NEXT: call void %fp() #[[#]]
define void @differing_values(i1 %c, ptr %fp) {
entry:
  br i1 %c, label %if, label %else

if:
  call void %fp() "nvvm.preserve_n_data"="8"
  br label %end

else:
  call void %fp() "nvvm.preserve_n_data"="2"
  br label %end

end:
  ret void
}

; Present on one side only: still must not be sunk.
; CHECK-LABEL: @one_sided(
; CHECK: if:
; CHECK-NEXT: call void %fp() #[[#]]
; CHECK: else:
; CHECK-NEXT: call void %fp(){{$}}
define void @one_sided(i1 %c, ptr %fp) {
entry:
  br i1 %c, label %if, label %else

if:
  call void %fp() "nvvm.preserve_n_data"="8"
  br label %end

else:
  call void %fp()
  br label %end

end:
  ret void
}

; Identical values: sinking is legal, and still happens. This is the control
; that shows the checks above are testing the attribute and not merely that
; sinking is disabled.
; CHECK-LABEL: @identical_values(
; CHECK: entry:
; CHECK: call void %fp() #[[#]]
; CHECK-NOT: call void %fp()
define void @identical_values(i1 %c, ptr %fp) {
entry:
  br i1 %c, label %if, label %else

if:
  call void %fp() "nvvm.preserve_n_data"="8"
  br label %end

else:
  call void %fp() "nvvm.preserve_n_data"="8"
  br label %end

end:
  ret void
}
