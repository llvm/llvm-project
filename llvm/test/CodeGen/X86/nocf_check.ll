; RUN: llc -mtriple=x86_64-unknown-unknown -x86-indirect-branch-tracking < %s | FileCheck %s

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; This test verify the handling of ''nocf_check'' attribute by the backend. ;;
;; The file was generated using the following C code:                        ;;
;;                                                                           ;;
;; void __attribute__((nocf_check)) NoCfCheckFunc(void) {}                   ;;
;;                                                                           ;;
;; typedef void(*FuncPointer)(void);                                         ;;
;; void NoCfCheckCall(FuncPointer f) {                                       ;;
;;   __attribute__((nocf_check)) FuncPointer p = f;                          ;;
;;   (*p)();                                                                 ;;
;; }                                                                         ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; Make sure that a function with ''nocf_check'' attribute is not instrumented
; with endbr instruction at the beginning.
define void @NoCfCheckFunc() #0 {
; CHECK-LABEL: NoCfCheckFunc
; CHECK-NOT:   endbr64
; CHECK:       retq
entry:
  ret void
}

; Make sure that notrack prefix is added before a call with ''nocf_check'' attribute.
define void @NoCfCheckCall(void ()* %f) {
; CHECK-LABEL: NoCfCheckCall
; CHECK:       notrack call
entry:
  %f.addr = alloca void ()*, align 4
  %p = alloca void ()*, align 4
  store void ()* %f, void ()** %f.addr, align 4
  %0 = load void ()*, void ()** %f.addr, align 4
  store void ()* %0, void ()** %p, align 4
  %1 = load void ()*, void ()** %p, align 4
  call void %1() #1
  ret void
}

attributes #0 = { noinline nocf_check nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nocf_check }

!llvm.module.flags = !{!0}

!0 = !{i32 4, !"cf-protection-branch", i32 1}
