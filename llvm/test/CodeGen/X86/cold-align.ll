;  Dont alter function alignment if marked cold
;
;  Cold attribute marks functions as also optimize for size. This normally collapses the
;  default function alignment. This can interfere with edit&continue effectiveness.
;
;
;  RUN:     llc -O2 <%s | FileCheck %s -check-prefixes TWO
;
;  TWO: .globl _ZN9Dismissed6ChillyEv
;  TWO-NEXT: .p2align 4, 0x90
;  TWO: .globl _ZN9Dismissed9TemparateEv
;  TWO-NEXT: .p2align 4, 0x90
;  TWO: .globl _ZN9Dismissed6SizzleEv
;  TWO-NEXT: .p2align 4, 0x90

; ModuleID = 'cold-align.cpp'
source_filename = "cold-align.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-sie-ps5"

; Function Attrs: cold mustprogress nofree norecurse nosync nounwind optsize sspstrong willreturn memory(none) uwtable
define hidden void @_ZN9Dismissed6ChillyEv(ptr nocapture noundef nonnull readnone align 1 dereferenceable(1) %this) local_unnamed_addr #0 align 2 {
entry:
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind sspstrong willreturn memory(none) uwtable
define hidden void @_ZN9Dismissed9TemparateEv(ptr nocapture noundef nonnull readnone align 1 dereferenceable(1) %this) local_unnamed_addr #1 align 2 {
entry:
  ret void
}

; Function Attrs: hot mustprogress nofree norecurse nosync nounwind sspstrong willreturn memory(none) uwtable
define hidden void @_ZN9Dismissed6SizzleEv(ptr nocapture noundef nonnull readnone align 1 dereferenceable(1) %this) local_unnamed_addr #2 align 2 {
entry:
  ret void
}

attributes #0 = { cold mustprogress nofree norecurse nosync nounwind optsize sspstrong willreturn memory(none) uwtable "denormal-fp-math"="preserve-sign,preserve-sign" "frame-pointer"="non-leaf" "keepalign"="true" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="znver2s" "target-features"="+adx,+aes,+avx,+avx2,+bmi,+bmi2,+clflushopt,+clwb,+clzero,+crc32,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+lwp,+lzcnt,+mmx,+movbe,+mwaitx,+pclmul,+popcnt,+prfchw,+rdpid,+rdpru,+rdrnd,+rdseed,+sahf,+sha,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+sse4a,+ssse3,+wbnoinvd,+x87,+xsave,+xsavec,+xsaveopt,+xsaves" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind sspstrong willreturn memory(none) uwtable "denormal-fp-math"="preserve-sign,preserve-sign" "frame-pointer"="non-leaf" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="znver2s" "target-features"="+adx,+aes,+avx,+avx2,+bmi,+bmi2,+clflushopt,+clwb,+clzero,+crc32,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+lwp,+lzcnt,+mmx,+movbe,+mwaitx,+pclmul,+popcnt,+prfchw,+rdpid,+rdpru,+rdrnd,+rdseed,+sahf,+sha,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+sse4a,+ssse3,+wbnoinvd,+x87,+xsave,+xsavec,+xsaveopt,+xsaves" }
attributes #2 = { hot mustprogress nofree norecurse nosync nounwind sspstrong willreturn memory(none) uwtable "denormal-fp-math"="preserve-sign,preserve-sign" "frame-pointer"="non-leaf" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="znver2s" "target-features"="+adx,+aes,+avx,+avx2,+bmi,+bmi2,+clflushopt,+clwb,+clzero,+crc32,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+lwp,+lzcnt,+mmx,+movbe,+mwaitx,+pclmul,+popcnt,+prfchw,+rdpid,+rdpru,+rdrnd,+rdseed,+sahf,+sha,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+sse4a,+ssse3,+wbnoinvd,+x87,+xsave,+xsavec,+xsaveopt,+xsaves" }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = !{i32 1, !"wchar_size", i32 2}
!1 = !{i32 1, !"SIE:STLVersion1", i32 1}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{i32 1, !"MaxTLSAlign", i32 256}
!6 = !{!"clang version 18.0.0"}
