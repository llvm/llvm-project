; RUN: llc -march=hexagon -enable-aa-sched-mi < %s
; REQUIRES: asserts

; Make sure the base is a register and not an address.

define fastcc void @Get_lsp_pol(i32* nocapture %f) #0 {
entry:
  %f5 = alloca i32, align 4
  %arrayidx103 = getelementptr inbounds i32, i32* %f, i32 4
  store i32 0, i32* %arrayidx103, align 4
  %f5.0.load185 = load volatile i32, i32* %f5, align 4
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
