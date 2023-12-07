; RUN: llc < %s -mtriple=aarch64-apple-ios -verify-machineinstrs

; This function tests that the machine verifier accepts an unconditional
; branch from an invoke basic block, to its EH landing pad basic block.
; The test is brittle and isn't ideally reduced, because in most cases the
; branch would be removed (for instance, turned into a fallthrough), and in
; that case, the machine verifier, which relies on analyzing branches for this
; kind of verification, is unable to check anything, so accepts the CFG.

define void @test_branch_to_landingpad() personality ptr @__objc_personality_v0 {
entry:
  br i1 undef, label %if.end50.thread, label %if.then6

lpad:
  %0 = landingpad { ptr, i32 }
          catch ptr @"OBJC_EHTYPE_$_NSString"
          catch ptr @OBJC_EHTYPE_id
          catch ptr null
  br i1 undef, label %invoke.cont33, label %catch.fallthrough

catch.fallthrough:
  %matches31 = icmp eq i32 undef, 0
  br i1 %matches31, label %invoke.cont41, label %finally.catchall

if.then6:
  invoke void @objc_exception_throw()
          to label %invoke.cont7 unwind label %lpad

invoke.cont7:
  unreachable

if.end50.thread:
  tail call void (ptr, ...) @printf(ptr @.str1, i32 125)
  tail call void (ptr, ...) @printf(ptr @.str1, i32 128)
  unreachable

invoke.cont33:
  tail call void (ptr, ...) @printf(ptr @.str1, i32 119)
  unreachable

invoke.cont41:
  invoke void @objc_exception_rethrow()
          to label %invoke.cont43 unwind label %lpad40

invoke.cont43:
  unreachable

lpad40:
  %1 = landingpad { ptr, i32 }
          catch ptr null
  br label %finally.catchall

finally.catchall:
  tail call void (ptr, ...) @printf(ptr @.str1, i32 125)
  unreachable
}

%struct._objc_typeinfo.12.129.194.285.350.493.519.532.571.597.623.765 = type { ptr, ptr, ptr }
%struct._class_t.10.127.192.283.348.491.517.530.569.595.621.764 = type { ptr, ptr, ptr, ptr, ptr }
%struct._objc_cache.0.117.182.273.338.481.507.520.559.585.611.754 = type opaque
%struct._class_ro_t.9.126.191.282.347.490.516.529.568.594.620.763 = type { i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct.__method_list_t.2.119.184.275.340.483.509.522.561.587.613.756 = type { i32, i32, [0 x %struct._objc_method.1.118.183.274.339.482.508.521.560.586.612.755] }
%struct._objc_method.1.118.183.274.339.482.508.521.560.586.612.755 = type { ptr, ptr, ptr }
%struct._objc_protocol_list.6.123.188.279.344.487.513.526.565.591.617.760 = type { i64, [0 x ptr] }
%struct._protocol_t.5.122.187.278.343.486.512.525.564.590.616.759 = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, ptr }
%struct._ivar_list_t.8.125.190.281.346.489.515.528.567.593.619.762 = type { i32, i32, [0 x %struct._ivar_t.7.124.189.280.345.488.514.527.566.592.618.761] }
%struct._ivar_t.7.124.189.280.345.488.514.527.566.592.618.761 = type { ptr, ptr, ptr, i32, i32 }
%struct._prop_list_t.4.121.186.277.342.485.511.524.563.589.615.758 = type { i32, i32, [0 x %struct._prop_t.3.120.185.276.341.484.510.523.562.588.614.757] }
%struct._prop_t.3.120.185.276.341.484.510.523.562.588.614.757 = type { ptr, ptr }

@.str1 = external unnamed_addr constant [17 x i8], align 1
@OBJC_EHTYPE_id = external global %struct._objc_typeinfo.12.129.194.285.350.493.519.532.571.597.623.765
@"OBJC_EHTYPE_$_NSString" = external global %struct._objc_typeinfo.12.129.194.285.350.493.519.532.571.597.623.765, section "__DATA,__datacoal_nt,coalesced", align 8

declare void @objc_exception_throw()
declare void @objc_exception_rethrow()
declare i32 @__objc_personality_v0(...)
declare void @printf(ptr nocapture readonly, ...)
