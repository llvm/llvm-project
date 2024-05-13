; RUN: llc -march=hexagon -O3 -verify-machineinstrs < %s
; REQUIRES: asserts

target triple = "hexagon"

%s.0 = type { ptr }
%s.1 = type { %s.2, ptr, i32, i32, i8, %s.3 }
%s.2 = type { ptr, i32 }
%s.3 = type { %s.4, %s.6, i32, i32 }
%s.4 = type { %s.5 }
%s.5 = type { i8 }
%s.6 = type { ptr, [12 x i8] }
%s.7 = type { %s.2, %s.8 }
%s.8 = type { ptr, ptr }
%s.9 = type { [16 x ptr] }
%s.10 = type { ptr, i32, i8, i8, i16, i32, i32, ptr, ptr, ptr }
%s.11 = type { ptr, i32, i32, ptr }
%s.12 = type { ptr, i32, ptr }

define i32 @f0() #0 personality ptr @f2 {
b0:
  %v0 = invoke dereferenceable(4) ptr @f1()
          to label %b1 unwind label %b2

b1:                                               ; preds = %b0
  %v1 = load i32, ptr undef, align 4
  %v2 = icmp eq i32 %v1, 0
  %v3 = zext i1 %v2 to i64
  %v4 = shl nuw nsw i64 %v3, 32
  %v5 = or i64 %v4, 0
  %v6 = call i64 @f3(ptr undef, i64 %v5, i64 4294967296, ptr nonnull dereferenceable(32) undef, ptr nonnull dereferenceable(1) undef, ptr nonnull dereferenceable(4) undef)
  unreachable

b2:                                               ; preds = %b0
  %v7 = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } undef
}

declare dereferenceable(4) ptr @f1()

declare i32 @f2(...)

declare i64 @f3(ptr nocapture readnone, i64, i64, ptr nocapture readonly dereferenceable(32), ptr nocapture dereferenceable(1), ptr nocapture dereferenceable(4)) unnamed_addr align 2

attributes #0 = { "target-cpu"="hexagonv55" }
