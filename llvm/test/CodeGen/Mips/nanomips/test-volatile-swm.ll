; RUN: llc -march=nanomips  < %s | FileCheck %s
target datalayout = "e-m:e-p:32:32-i8:8:32-i16:16:32-i64:64-n32:64-S128"
target triple = "nanomips"

%struct.S = type { i32, i32 }

declare dso_local inreg { i32, i32 } @GetStruct(...) local_unnamed_addr

define dso_local void @test(%struct.S* %p, i32 signext %i) local_unnamed_addr {
entry:
; CHECK-NOT: swm
; CHECK: sw
; CHECK: sw
  %call = tail call inreg { i32, i32 } bitcast ({ i32, i32 } (...)* @GetStruct to { i32, i32 } ()*)() 
  %0 = extractvalue { i32, i32 } %call, 0
  %1 = extractvalue { i32, i32 } %call, 1
  %B1 = getelementptr inbounds %struct.S, %struct.S* %p, i32 0, i32 1
  store volatile i32 %1, i32* %B1, align 4
  %A26 = bitcast %struct.S* %p to i32*
  store volatile i32 %0, i32* %A26, align 4
  ret void
}

