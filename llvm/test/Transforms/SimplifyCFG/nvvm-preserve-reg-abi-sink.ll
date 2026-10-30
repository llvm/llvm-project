; RUN: opt -S -passes='simplifycfg<sink-common-insts>' < %s | FileCheck %s

; CHECK-LABEL: @differing_values(
; CHECK: if:
; CHECK-NEXT: call void %fp() #[[#]]
; CHECK: else:
; CHECK-NEXT: call void %fp() #[[#]]
define void @differing_values(i1 %c, ptr %fp) {
entry:
  br i1 %c, label %if, label %else

if:
  call void %fp() "nvvm.abi_preserve"="8"
  br label %end

else:
  call void %fp() "nvvm.abi_preserve"="2"
  br label %end

end:
  ret void
}

; CHECK-LABEL: @one_sided(
; CHECK: if:
; CHECK-NEXT: call void %fp() #[[#]]
; CHECK: else:
; CHECK-NEXT: call void %fp(){{$}}
define void @one_sided(i1 %c, ptr %fp) {
entry:
  br i1 %c, label %if, label %else

if:
  call void %fp() "nvvm.abi_preserve"="8"
  br label %end

else:
  call void %fp()
  br label %end

end:
  ret void
}

; CHECK-LABEL: @identical_values(
; CHECK: entry:
; CHECK: call void %fp() #[[#]]
; CHECK-NOT: call void %fp()
define void @identical_values(i1 %c, ptr %fp) {
entry:
  br i1 %c, label %if, label %else

if:
  call void %fp() "nvvm.abi_preserve"="8"
  br label %end

else:
  call void %fp() "nvvm.abi_preserve"="8"
  br label %end

end:
  ret void
}
