; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -start-after=codegenprepare -stop-before finalize-isel | FileCheck %s

define void @_Z3fooiPiPs(<8 x i32> %gepload, <8 x i1> %0) #0 {
entry:
  %1 = trunc <8 x i32> %gepload to <8 x i16>
  tail call void @llvm.masked.compressstore.v8i16(<8 x i16> %1, ptr null, <8 x i1> %0)
  ret void
}

; CHECK-LABEL: bb.0.entry:
; CHECK:         %1:vr128x = COPY $xmm1
; CHECK-NEXT:    %0:vr256x = COPY $ymm0
; CHECK-NEXT:    %2:vr128x = VPSLLWZ128ri %1, 15
; CHECK-NEXT:    %3:vk16wm = VPMOVW2MZ128rr killed %2
; CHECK-NEXT:    %4:vr128x = VPMOVDWZ256rr %0
; CHECK-NEXT:    VPCOMPRESSWZ128mrk $noreg, 1, $noreg, 0, $noreg, killed %3, killed %4 :: (store unknown-size into `ptr null`, align 16)
; CHECK-NEXT:    RET 0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.masked.compressstore.v8i16(<8 x i16>, ptr nocapture, <8 x i1>) #1

attributes #0 = { "target-cpu"="icelake-server" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
