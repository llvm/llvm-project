; RUN: llc -mtriple=aarch64-apple-ios -global-isel -stop-after=irtranslator %s -o - | FileCheck %s

%struct.foo = type { i64, i64, %struct.pluto, %struct.pluto }
%struct.pluto = type { %struct.wombat }
%struct.wombat = type { ptr, ptr, %struct.barney }
%struct.barney = type { %struct.widget }
%struct.widget = type { ptr }

declare i32 @hoge(...)

define void @pluto() align 2 personality ptr @hoge {
; CHECK-LABEL: @pluto
; CHECK: bb.1.bb
; CHECK: successors: %bb.2(0x00000000), %bb.3(0x80000000)
; CHECK: EH_LABEL <mcsymbol >
; CHECK: G_BR %bb.2

bb:
  invoke void @spam()
          to label %bb1 unwind label %bb2

bb1:                                              ; preds = %bb
  unreachable

bb2:                                              ; preds = %bb
  %tmp = landingpad { ptr, i32 }
          cleanup
  %tmp3 = getelementptr inbounds %struct.foo, ptr undef, i64 0, i32 3, i32 0, i32 0
  resume { ptr, i32 } %tmp
}

declare void @spam()
