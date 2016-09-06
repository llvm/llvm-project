; RUN: llc -march=hexagon < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

; Generate the inline restore for single register pair for functions
; with "optsize" attribute.

; CHECK-LABEL:    fred_os
; CHECK-DAG:      memd{{.*}} = r17:16
; CHECK-DAG:      r17:16 = memd{{.*}}
; CHECK-DAG:      deallocframe
; CHECK-NOT:  call __restore

define i32 @fred_os(i32 %x) #0 {
entry:
  %call = tail call i32 @foo(i32 %x) #2
  %call1 = tail call i32 @bar(i32 %x, i32 %call) #2
  ret i32 %call1
}

; Generate the restoring call for single register pair for functions
; with "minsize" attribute.

; CHECK-LABEL:    fred_oz
; CHECK-DAG:      memd{{.*}} = r17:16
; CHECK-NOT:  r17:16 = memd{{.*}}
; CHECK-DAG:      call __restore

define i32 @fred_oz(i32 %x) #1 {
entry:
  %call = tail call i32 @foo(i32 %x) #2
  %call1 = tail call i32 @bar(i32 %x, i32 %call) #2
  ret i32 %call1
}

declare i32 @foo(i32) #2
declare i32 @bar(i32, i32) #2

attributes #0 = { nounwind optsize "disable-tail-calls"="false" }
attributes #1 = { nounwind minsize "disable-tail-calls"="false" }
attributes #2 = { nounwind optsize }
