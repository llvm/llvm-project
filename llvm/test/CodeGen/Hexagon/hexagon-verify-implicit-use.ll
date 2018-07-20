; RUN: llc -march=hexagon -O3 -verify-machineinstrs < %s
; REQUIRES: asserts

target triple = "hexagon"

%s.0 = type { %s.1* }
%s.1 = type { %s.2, %s.2**, i32, i32, i8, %s.3 }
%s.2 = type { i32 (...)**, i32 }
%s.3 = type { %s.4, %s.6, i32, i32 }
%s.4 = type { %s.5 }
%s.5 = type { i8 }
%s.6 = type { i8*, [12 x i8] }
%s.7 = type { %s.2, %s.8 }
%s.8 = type { %s.9*, %s.9* }
%s.9 = type { [16 x i16*] }
%s.10 = type { i32 (...)**, i32, i8, i8, i16, i32, i32, %s.11*, %s.12*, %s.0* }
%s.11 = type { %s.11*, i32, i32, i8* }
%s.12 = type { %s.12*, i32, void (i8, %s.10*, i32)* }

define i32 @f0() #0 personality i8* bitcast (i32 (...)* @f2 to i8*) {
b0:
  %v0 = invoke dereferenceable(4) %s.0* @f1()
          to label %b1 unwind label %b2

b1:                                               ; preds = %b0
  %v1 = load i32, i32* undef, align 4
  %v2 = icmp eq i32 %v1, 0
  %v3 = zext i1 %v2 to i64
  %v4 = shl nuw nsw i64 %v3, 32
  %v5 = or i64 %v4, 0
  %v6 = call i64 @f3(%s.7* undef, i64 %v5, i64 4294967296, %s.10* nonnull dereferenceable(32) undef, i8* nonnull dereferenceable(1) undef, i32* nonnull dereferenceable(4) undef)
  unreachable

b2:                                               ; preds = %b0
  %v7 = landingpad { i8*, i32 }
          cleanup
  resume { i8*, i32 } undef
}

declare dereferenceable(4) %s.0* @f1()

declare i32 @f2(...)

declare i64 @f3(%s.7* nocapture readnone, i64, i64, %s.10* nocapture readonly dereferenceable(32), i8* nocapture dereferenceable(1), i32* nocapture dereferenceable(4)) unnamed_addr align 2

attributes #0 = { "target-cpu"="hexagonv55" }
