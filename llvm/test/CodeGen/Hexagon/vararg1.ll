; RUN: llc -march=hexagon -mcpu=hexagonv73 -mattr=+hvx,+hvx-length64b -disable-packetizer -enable-local-pullup=false < %s | FileCheck %s
; Disabling the packetizer and local pullup to make it easier to check instruction ordering
; (argument setup before calls).

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

@.str.1 = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
@str = private unnamed_addr constant [4 x i8] c"foo\00"

define i32 @bar(i32 %x, i32 %y) #0 {
entry:
  %mul = mul nsw i32 %y, %x
  %add = add nsw i32 %mul, 1
  ret i32 %add
}

define void @foo(...) #1 {
entry:
  %puts = tail call i32 @puts(ptr getelementptr inbounds ([4 x i8], [4 x i8]* @str, i32 0, i32 0))
  ret void
}


; CHECK-LABEL: fred
; CHECK: ##.L.str.1
; CHECK: memw(r29+#0) = #99
; CHECK: callr

; CHECK: ##.L.str.1
; CHECK: memw(r29+#0) = #99
; CHECK: call printf

; CHECK-DAG: memw(r29+#8) =
; CHECK-DAG: memw(r29+#4) = #2
; CHECK-DAG: memw(r29+#0) = #1
; CHECK: call foo

; CHECK: r1:0 = combine(#5,#6)
; CHECK: call bar

define i32 @fred(ptr nocapture %printf_ptr) {
entry:
  %call = tail call i32 (ptr, ...) %printf_ptr(ptr getelementptr inbounds ([4 x i8], [4 x i8]* @.str.1, i32 0, i32 0), i32 99)
  %call1 = tail call i32 (ptr, ...) @printf(ptr getelementptr inbounds ([4 x i8], [4 x i8]* @.str.1, i32 0, i32 0), i32 99) #5
  tail call void (...) @foo(i32 1, i32 2, i32 3)
  %call2 = tail call i32 @bar(i32 6, i32 5)
  %add = add nsw i32 %call2, 1
  ret i32 %add
}

define i32 @main() #4 {
entry:
  %call = tail call i32 @fred(ptr nonnull @printf)
  ret i32 %call
}

declare i32 @puts(ptr nocapture) #5
declare i32 @printf(ptr nocapture readonly, ...) #2

attributes #0 = { norecurse nounwind readnone }
attributes #1 = { nounwind }
attributes #2 = { nounwind }
attributes #4 = { norecurse }
attributes #5 = { nounwind }
