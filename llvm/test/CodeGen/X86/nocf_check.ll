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
;;   NoCfCheckFunc();                                                        ;;
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

; Ensure the notrack prefix is added before an indirect call using a pointer
; with ''nocf_check'' attribute. Also ensure a direct call to a function with
; the ''nocf_check'' attribute is correctly generated without notrack prefix.
define void @NoCfCheckCall(void ()* %f) #1 {
; CHECK-LABEL: NoCfCheckCall
; CHECK:       notrack call
; CHECK:       callq NoCfCheckFunc
entry:
  %f.addr = alloca void ()*, align 4
  %p = alloca void ()*, align 4
  store void ()* %f, void ()** %f.addr, align 4
  %0 = load void ()*, void ()** %f.addr, align 4
  store void ()* %0, void ()** %p, align 4
  %1 = load void ()*, void ()** %p, align 4
  call void %1() #2
	call void @NoCfCheckFunc() #2
  ret void
}

attributes #0 = { nocf_check noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nocf_check }

!llvm.module.flags = !{!0}

!0 = !{i32 8, !"cf-protection-branch", i32 1}
