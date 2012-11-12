; RUN: opt -O2 -S %s | FileCheck %s

; Run global DCE to eliminate unused ctor and dtor.
; rdar://9142819

; CHECK: main
; CHECK-NOT: _ZN4BaseC1Ev
; CHECK-NOT: _ZN4BaseD1Ev
; CHECK-NOT: _ZN4BaseD2Ev
; CHECK-NOT: _ZN4BaseC2Ev
; CHECK-NOT: _ZN4BaseD0Ev

%class.Base = type { i32 (...)** }

@_ZTV4Base = linkonce_odr unnamed_addr constant [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI4Base to i8*), i8* bitcast (void (%class.Base*)* @_ZN4BaseD1Ev to i8*), i8* bitcast (void (%class.Base*)* @_ZN4BaseD0Ev to i8*)]
@_ZTVN10__cxxabiv117__class_type_infoE = external global i8*
@_ZTS4Base = linkonce_odr constant [6 x i8] c"4Base\00"
@_ZTI4Base = linkonce_odr unnamed_addr constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([6 x i8]* @_ZTS4Base, i32 0, i32 0) }

define i32 @main() uwtable ssp {
entry:
  %retval = alloca i32, align 4
  %b = alloca %class.Base, align 8
  %cleanup.dest.slot = alloca i32
  store i32 0, i32* %retval
  call void @_ZN4BaseC1Ev(%class.Base* %b)
  store i32 0, i32* %retval
  store i32 1, i32* %cleanup.dest.slot
  call void @_ZN4BaseD1Ev(%class.Base* %b)
  %0 = load i32* %retval
  ret i32 %0
}

define linkonce_odr void @_ZN4BaseC1Ev(%class.Base* %this) unnamed_addr uwtable ssp align 2 {
entry:
  %this.addr = alloca %class.Base*, align 8
  store %class.Base* %this, %class.Base** %this.addr, align 8
  %this1 = load %class.Base** %this.addr
  call void @_ZN4BaseC2Ev(%class.Base* %this1)
  ret void
}

define linkonce_odr void @_ZN4BaseD1Ev(%class.Base* %this) unnamed_addr uwtable ssp align 2 {
entry:
  %this.addr = alloca %class.Base*, align 8
  store %class.Base* %this, %class.Base** %this.addr, align 8
  %this1 = load %class.Base** %this.addr
  call void @_ZN4BaseD2Ev(%class.Base* %this1)
  ret void
}

define linkonce_odr void @_ZN4BaseD2Ev(%class.Base* %this) unnamed_addr nounwind uwtable ssp align 2 {
entry:
  %this.addr = alloca %class.Base*, align 8
  store %class.Base* %this, %class.Base** %this.addr, align 8
  %this1 = load %class.Base** %this.addr
  ret void
}

define linkonce_odr void @_ZN4BaseC2Ev(%class.Base* %this) unnamed_addr nounwind uwtable ssp align 2 {
entry:
  %this.addr = alloca %class.Base*, align 8
  store %class.Base* %this, %class.Base** %this.addr, align 8
  %this1 = load %class.Base** %this.addr
  %0 = bitcast %class.Base* %this1 to i8***
  store i8** getelementptr inbounds ([4 x i8*]* @_ZTV4Base, i64 0, i64 2), i8*** %0
  ret void
}

define linkonce_odr void @_ZN4BaseD0Ev(%class.Base* %this) unnamed_addr uwtable ssp align 2 {
entry:
  %this.addr = alloca %class.Base*, align 8
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  store %class.Base* %this, %class.Base** %this.addr, align 8
  %this1 = load %class.Base** %this.addr
  invoke void @_ZN4BaseD1Ev(%class.Base* %this1)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  %0 = bitcast %class.Base* %this1 to i8*
  call void @_ZdlPv(i8* %0) nounwind
  ret void

lpad:                                             ; preds = %entry
  %1 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          cleanup
  %2 = extractvalue { i8*, i32 } %1, 0
  store i8* %2, i8** %exn.slot
  %3 = extractvalue { i8*, i32 } %1, 1
  store i32 %3, i32* %ehselector.slot
  %4 = bitcast %class.Base* %this1 to i8*
  call void @_ZdlPv(i8* %4) nounwind
  br label %eh.resume

eh.resume:                                        ; preds = %lpad
  %exn = load i8** %exn.slot
  %sel = load i32* %ehselector.slot
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn, 0
  %lpad.val2 = insertvalue { i8*, i32 } %lpad.val, i32 %sel, 1
  resume { i8*, i32 } %lpad.val2
}

declare i32 @__gxx_personality_v0(...)

declare void @_ZdlPv(i8*) nounwind
