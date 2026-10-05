; RUN: llc -mtriple=x86_64-pc-windows-msvc -emit-call-site-info \
; RUN:   -stop-after=finalize-isel < %s | FileCheck %s

; With import call optimization, the callee of an indirect call has to be in
; RAX, so a copy comes before the call. A poison callee gets an IMPLICIT_DEF
; there instead. The call site info, nomerge and the heap alloc marker belong
; to the call.

; CHECK-LABEL: name: call
; CHECK:       callSites: []
; CHECK:         {{%[0-9]+}}:gr64_a = nomerge COPY
; CHECK-NEXT:    {{^ +}}CALL64r_ImpCall {{.*}}, implicit-def $ssp{{$}}
define void @call(ptr %fp) {
  call void %fp(i32 1) #0, !heapallocsite !1
  ret void
}

; CHECK-LABEL: name: tail_call
; CHECK:       callSites: []
; CHECK:         {{%[0-9]+}}:gr64_a = nomerge COPY
; CHECK-NEXT:    {{^ +}}TCRETURNri64_ImpCall
define void @tail_call(ptr %fp) {
  tail call void %fp(i32 1) #0
  ret void
}

; CHECK-LABEL: name: poison_callee
; CHECK:       callSites: []
; CHECK:         {{%[0-9]+}}:gr64_a = nomerge IMPLICIT_DEF
; CHECK-NEXT:    {{^ +}}CALL64r_ImpCall
define void @poison_callee() {
  call void poison(i32 1) #0
  ret void
}

attributes #0 = { nomerge }

!llvm.module.flags = !{!0, !2}
!0 = !{i32 1, !"import-call-optimization", i32 1}
!1 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!2 = !{i32 2, !"Debug Info Version", i32 3}
