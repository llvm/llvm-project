; RUN: opt < %s -basicaa -instcombine -inline -functionattrs -licm -simple-loop-unswitch -gvn -verify
; PR12573
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.7.0"

%class.D.22.42.66.102.138.158.178.198.238.242.246.250.262.294.302.338.346.379 = type { %class.C.23.43.67.103.139.159.179.199.239.243.247.251.263.295.303.339.347.376*, %class.B.21.41.65.101.137.157.177.197.237.241.245.249.261.293.301.337.345.378 }
%class.C.23.43.67.103.139.159.179.199.239.243.247.251.263.295.303.339.347.376 = type { %class.D.22.42.66.102.138.158.178.198.238.242.246.250.262.294.302.338.346.379* }
%class.B.21.41.65.101.137.157.177.197.237.241.245.249.261.293.301.337.345.378 = type { %class.A.20.40.64.100.136.156.176.196.236.240.244.248.260.292.300.336.344.377* }
%class.A.20.40.64.100.136.156.176.196.236.240.244.248.260.292.300.336.344.377 = type { i8 }

define void @_Z23get_reconstruction_pathv() uwtable ssp personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %c = alloca %class.D.22.42.66.102.138.158.178.198.238.242.246.250.262.294.302.338.346.379, align 8
  br label %for.cond

for.cond:                                         ; preds = %for.end, %entry
  invoke void @_ZN1DptEv(%class.D.22.42.66.102.138.158.178.198.238.242.246.250.262.294.302.338.346.379* %c)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %for.cond
  invoke void @_ZN1C3endEv()
          to label %for.cond3 unwind label %lpad

for.cond3:                                        ; preds = %invoke.cont6, %invoke.cont
  invoke void @_ZN1DptEv(%class.D.22.42.66.102.138.158.178.198.238.242.246.250.262.294.302.338.346.379* %c)
          to label %invoke.cont4 unwind label %lpad

invoke.cont4:                                     ; preds = %for.cond3
  invoke void @_ZN1C3endEv()
          to label %invoke.cont6 unwind label %lpad

invoke.cont6:                                     ; preds = %invoke.cont4
  br i1 undef, label %for.cond3, label %for.end

lpad:                                             ; preds = %for.end, %invoke.cont4, %for.cond3, %invoke.cont, %for.cond
  %0 = landingpad { i8*, i32 }
          cleanup
  resume { i8*, i32 } undef

for.end:                                          ; preds = %invoke.cont6
  invoke void @_ZN1C13_M_insert_auxER1D()
          to label %for.cond unwind label %lpad
}

define void @_ZN1DptEv(%class.D.22.42.66.102.138.158.178.198.238.242.246.250.262.294.302.338.346.379* %this) uwtable ssp align 2 {
entry:
  %this.addr = alloca %class.D.22.42.66.102.138.158.178.198.238.242.246.250.262.294.302.338.346.379*, align 8
  store %class.D.22.42.66.102.138.158.178.198.238.242.246.250.262.294.302.338.346.379* %this, %class.D.22.42.66.102.138.158.178.198.238.242.246.250.262.294.302.338.346.379** %this.addr, align 8
  %this1 = load %class.D.22.42.66.102.138.158.178.198.238.242.246.250.262.294.302.338.346.379*, %class.D.22.42.66.102.138.158.178.198.238.242.246.250.262.294.302.338.346.379** %this.addr
  %px = getelementptr inbounds %class.D.22.42.66.102.138.158.178.198.238.242.246.250.262.294.302.338.346.379, %class.D.22.42.66.102.138.158.178.198.238.242.246.250.262.294.302.338.346.379* %this1, i32 0, i32 0
  %0 = load %class.C.23.43.67.103.139.159.179.199.239.243.247.251.263.295.303.339.347.376*, %class.C.23.43.67.103.139.159.179.199.239.243.247.251.263.295.303.339.347.376** %px, align 8
  %tobool = icmp ne %class.C.23.43.67.103.139.159.179.199.239.243.247.251.263.295.303.339.347.376* %0, null
  br i1 %tobool, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  call void @_Z10__assert13v() noreturn
  unreachable

cond.end:                                         ; preds = %entry
  ret void
}

declare i32 @__gxx_personality_v0(...)

declare void @_ZN1C3endEv()

define void @_ZN1C13_M_insert_auxER1D() uwtable ssp align 2 {
entry:
  ret void
}

define void @_ZN1DD1Ev() unnamed_addr uwtable inlinehint ssp align 2 {
entry:
  ret void
}

define void @_ZN1DD2Ev() unnamed_addr uwtable inlinehint ssp align 2 {
entry:
  ret void
}

define void @_ZN1BD1Ev() unnamed_addr uwtable ssp align 2 {
entry:
  ret void
}

define void @_ZN1BD2Ev() unnamed_addr uwtable ssp align 2 {
entry:
  ret void
}

define void @_ZN1BaSERS_() uwtable ssp align 2 {
entry:
  unreachable
}

declare void @_Z10__assert13v() noreturn
