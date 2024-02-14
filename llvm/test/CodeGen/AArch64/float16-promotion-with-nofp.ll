; RUN: llc -mcpu=cortex-r82 -O1 -o - %s | FileCheck %s

; Source used:
; __fp16 f2h(float a) { return a; }
; Compiled with: clang --target=aarch64-arm-none-eabi -march=armv8-r+nofp

define hidden noundef nofpclass(nan inf) half @f2h(float noundef nofpclass(nan inf) %a) local_unnamed_addr #0 {
;CHECK:      f2h:                                    // @f2h
;CHECK-NEXT: // %bb.0:                               // %entry
;CHECK-NEXT:     str x30, [sp, #-16]!                // 8-byte Folded Spill
;CHECK-NEXT:     bl  __gnu_h2f_ieee
;CHECK-NEXT:     ldr x30, [sp], #16                  // 8-byte Folded Reload
;CHECK-NEXT:     ret
entry:
  %0 = fptrunc float %a to half
  ret half %0
}


attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) "denormal-fp-math"="preserve-sign,preserve-sign" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+crc,+lse,+pauth,+ras,+rcpc,+sb,+ssbs,+v8r,-complxnum,-dotprod,-fmv,-fp-armv8,-fp16fml,-fullfp16,-jsconv,-neon,-rdm" }

