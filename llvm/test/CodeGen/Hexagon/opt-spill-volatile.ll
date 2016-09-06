; RUN: llc -march=hexagon < %s | FileCheck %s
; Check that the load/store to the volatile stack object has not been
; optimized away.

target triple = "hexagon"

; CHECK-LABEL: foo
; CHECK: memw(r29+#4) =
; CHECK: = memw(r29 + #4)
define i32 @foo(i32 %a) #0 {
entry:
  %x = alloca i32, align 4
  %x.0.x.0..sroa_cast = bitcast i32* %x to i8*
  call void @llvm.lifetime.start(i64 4, i8* %x.0.x.0..sroa_cast)
  store volatile i32 0, i32* %x, align 4
  %call = tail call i32 bitcast (i32 (...)* @bar to i32 ()*)() #0
  %x.0.x.0. = load volatile i32, i32* %x, align 4
  %add = add nsw i32 %x.0.x.0., %a
  call void @llvm.lifetime.end(i64 4, i8* %x.0.x.0..sroa_cast)
  ret i32 %add
}

declare void @llvm.lifetime.start(i64, i8* nocapture) #1
declare void @llvm.lifetime.end(i64, i8* nocapture) #1

declare i32 @bar(...) #0

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }
