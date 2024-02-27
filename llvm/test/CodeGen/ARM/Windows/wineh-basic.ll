; RUN: llc < %s | FileCheck %s

; CHECK: "??1field@@AAA@XZ":
; CHECK: bl free

; C++ source:
; class field { ~field(); };
; extern "C" void free(void *ptr);
; field::~field() { free((void *)0); }

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7--windows-msvc19.0.24210"

%class.field = type { i8 }

; Function Attrs: nounwind
define arm_aapcs_vfpcc void @"\01??1field@@AAA@XZ"(ptr nocapture readnone %this) unnamed_addr #0 align 2 personality ptr @__CxxFrameHandler3 {
entry:
  invoke arm_aapcs_vfpcc void @free(ptr null)
          to label %invoke.cont unwind label %terminate

invoke.cont:                                      ; preds = %entry
  ret void

terminate:                                        ; preds = %entry
  %0 = cleanuppad within none []
  tail call arm_aapcs_vfpcc void @__std_terminate() #2 [ "funclet"(token %0) ]
  unreachable
}

declare arm_aapcs_vfpcc void @free(ptr) local_unnamed_addr #1

declare arm_aapcs_vfpcc i32 @__CxxFrameHandler3(...)

declare arm_aapcs_vfpcc void @__std_terminate() local_unnamed_addr

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="cortex-a9" "target-features"="+dsp,+fp16,+neon,+strict-align,+vfp3" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="cortex-a9" "target-features"="+dsp,+fp16,+neon,+strict-align,+vfp3" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 2}
!1 = !{i32 1, !"min_enum_size", i32 4}
!2 = !{!"clang version 4.0.0 (trunk 284595) (llvm/trunk 284597)"}
