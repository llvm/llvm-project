; RUN: opt -O2 -S < %s | FileCheck %s

; Run global DCE to eliminate unused ctor and dtor.
; rdar://9142819

; CHECK: main
; CHECK-NOT: _ZN4BaseC1Ev
; CHECK-NOT: _ZN4BaseD1Ev
; CHECK-NOT: _ZN4BaseD2Ev
; CHECK-NOT: _ZN4BaseC2Ev
; CHECK-NOT: _ZN4BaseD0Ev

%class.Base = type { ptr }

@_ZTV4Base = linkonce_odr unnamed_addr constant [4 x ptr] [ptr null, ptr @_ZTI4Base, ptr @_ZN4BaseD1Ev, ptr @_ZN4BaseD0Ev]
@_ZTVN10__cxxabiv117__class_type_infoE = external global ptr
@_ZTS4Base = linkonce_odr constant [6 x i8] c"4Base\00"
@_ZTI4Base = linkonce_odr unnamed_addr constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS4Base }

define i32 @main() uwtable ssp {
entry:
  %retval = alloca i32, align 4
  %b = alloca %class.Base, align 8
  %cleanup.dest.slot = alloca i32
  store i32 0, ptr %retval
  call void @_ZN4BaseC1Ev(ptr %b)
  store i32 0, ptr %retval
  store i32 1, ptr %cleanup.dest.slot
  call void @_ZN4BaseD1Ev(ptr %b)
  %0 = load i32, ptr %retval
  ret i32 %0
}

define linkonce_odr void @_ZN4BaseC1Ev(ptr %this) unnamed_addr uwtable ssp align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr
  call void @_ZN4BaseC2Ev(ptr %this1)
  ret void
}

define linkonce_odr void @_ZN4BaseD1Ev(ptr %this) unnamed_addr uwtable ssp align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr
  call void @_ZN4BaseD2Ev(ptr %this1)
  ret void
}

define linkonce_odr void @_ZN4BaseD2Ev(ptr %this) unnamed_addr nounwind uwtable ssp align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr
  ret void
}

define linkonce_odr void @_ZN4BaseC2Ev(ptr %this) unnamed_addr nounwind uwtable ssp align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr
  store ptr getelementptr inbounds ([4 x ptr], ptr @_ZTV4Base, i64 0, i64 2), ptr %this1
  ret void
}

define linkonce_odr void @_ZN4BaseD0Ev(ptr %this) unnamed_addr uwtable ssp align 2 personality ptr @__gxx_personality_v0 {
entry:
  %this.addr = alloca ptr, align 8
  %exn.slot = alloca ptr
  %ehselector.slot = alloca i32
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr
  invoke void @_ZN4BaseD1Ev(ptr %this1)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  call void @_ZdlPv(ptr %this1) nounwind
  ret void

lpad:                                             ; preds = %entry
  %0 = landingpad { ptr, i32 }
          cleanup
  %1 = extractvalue { ptr, i32 } %0, 0
  store ptr %1, ptr %exn.slot
  %2 = extractvalue { ptr, i32 } %0, 1
  store i32 %2, ptr %ehselector.slot
  call void @_ZdlPv(ptr %this1) nounwind
  br label %eh.resume

eh.resume:                                        ; preds = %lpad
  %exn = load ptr, ptr %exn.slot
  %sel = load i32, ptr %ehselector.slot
  %lpad.val = insertvalue { ptr, i32 } undef, ptr %exn, 0
  %lpad.val2 = insertvalue { ptr, i32 } %lpad.val, i32 %sel, 1
  resume { ptr, i32 } %lpad.val2
}

declare i32 @__gxx_personality_v0(...)

declare void @_ZdlPv(ptr) nounwind
