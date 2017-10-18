; RUN: llc -march=hexagon < %s | FileCheck %s
; Check that all the offsets in the .cfi_offset instructions are negative.
; They are all based on R30+8 which points to the pair FP/LR stored by an
; allocframe. Since the stack grows towards negative addresses, anything
; in the current stack frame will have a negative offset with respect to
; R30+8.

; CHECK: cfi_def_cfa r30
; CHECK-NOT: .cfi_offset r{{[0-9]+}}, {{[^-]}}

target triple = "hexagon"

define i64 @_Z3fooxxx(i64 %x, i64 %y, i64 %z) #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %call = invoke i64 @_Z3barxxx(i64 %x, i64 %y, i64 %z)
          to label %try.cont unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          catch i8* null
  %1 = extractvalue { i8*, i32 } %0, 0
  %2 = tail call i8* @__cxa_begin_catch(i8* %1) #1
  tail call void @__cxa_end_catch()
  br label %try.cont

try.cont:                                         ; preds = %entry, %lpad
  %a.0 = phi i64 [ 0, %lpad ], [ %call, %entry ]
  %mul = mul nsw i64 %y, %x
  %sub = sub i64 %mul, %z
  %add = add nsw i64 %sub, %a.0
  ret i64 %add
}

declare i64 @_Z3barxxx(i64, i64, i64) #0

declare i32 @__gxx_personality_v0(...)

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()

attributes #0 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv60" "target-features"="-hvx" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
