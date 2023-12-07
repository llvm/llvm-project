; RUN: llc -mtriple=aarch64-windows %s -o - | FileCheck %s

define hidden swifttailcc void @test(ptr noalias nocapture %0, ptr swiftasync %1, ptr %2, ptr noalias nocapture %3, ptr nocapture dereferenceable(8) %4, ptr %5, ptr %6, ptr %Act, ptr %Err, ptr %Res, ptr %Act.DistributedActor, ptr %Err.Error, ptr %Res.Decodable, ptr %Res.Encodable, ptr swiftself %7) #0 {
entry:
  ret void
}

; Check that there is no .seh_endprologue but there is seh_startepilogue/seh_endepilogue.
; CHECK-NOT: .seh_endprologue
; CHECK:     .seh_startepilogue
; CHECK:     add sp, sp, #48
; CHECK:     .seh_stackalloc 48
; CHECK:     .seh_endepilogue
