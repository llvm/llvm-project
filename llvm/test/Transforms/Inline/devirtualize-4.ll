; RUN: opt < %s -passes='cgscc(devirt<4>(inline)),function(sroa,early-cse)' -S | FileCheck %s
; RUN: opt < %s -passes='default<O3>' -S | FileCheck %s

; Check that DoNotOptimize is inlined into Test.
; CHECK: @_Z4Testv()
; CHECK-NOT: ret void
; CHECK: call void asm
; CHECK: ret void

;template <class T>
;void DoNotOptimize(const T& var) {
;  asm volatile("" : "+m"(const_cast<T&>(var)));
;}
;
;class Interface {
; public:
;  virtual void Run() = 0;
;};
;
;class Impl : public Interface {
; public:
;  Impl() : f(3) {}
;  void Run() { DoNotOptimize(this); }
;
; private:
;  int f;
;};
;
;static void IndirectRun(Interface& o) { o.Run(); }
;
;void Test() {
;  Impl o;
;  IndirectRun(o);
;}

%class.Impl = type <{ %class.Interface, i32, [4 x i8] }>
%class.Interface = type { ptr }

@_ZTV4Impl = linkonce_odr dso_local unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI4Impl, ptr @_ZN4Impl3RunEv] }, align 8
@_ZTVN10__cxxabiv120__si_class_type_infoE = external dso_local global ptr
@_ZTS4Impl = linkonce_odr dso_local constant [6 x i8] c"4Impl\00", align 1
@_ZTVN10__cxxabiv117__class_type_infoE = external dso_local global ptr
@_ZTS9Interface = linkonce_odr dso_local constant [11 x i8] c"9Interface\00", align 1
@_ZTI9Interface = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS9Interface }, align 8
@_ZTI4Impl = linkonce_odr dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTS4Impl, ptr @_ZTI9Interface }, align 8
@_ZTV9Interface = linkonce_odr dso_local unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI9Interface, ptr @__cxa_pure_virtual] }, align 8

define dso_local void @_Z4Testv() local_unnamed_addr {
entry:
  %o = alloca %class.Impl, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %o)
  call void @_ZN4ImplC2Ev(ptr nonnull %o)
  call fastcc void @_ZL11IndirectRunR9Interface(ptr nonnull dereferenceable(8) %o)
  call void @llvm.lifetime.end.p0(ptr nonnull %o)
  ret void
}

declare void @llvm.lifetime.start.p0(ptr nocapture)

define linkonce_odr dso_local void @_ZN4ImplC2Ev(ptr %this) unnamed_addr align 2 {
entry:
  call void @_ZN9InterfaceC2Ev(ptr %this)
  store ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr] }, ptr @_ZTV4Impl, i64 0, i32 0, i64 2), ptr %this, align 8
  %f = getelementptr inbounds %class.Impl, ptr %this, i64 0, i32 1
  store i32 3, ptr %f, align 8
  ret void
}

define internal fastcc void @_ZL11IndirectRunR9Interface(ptr dereferenceable(8) %o) unnamed_addr {
entry:
  %vtable = load ptr, ptr %o, align 8
  %0 = load ptr, ptr %vtable, align 8
  call void %0(ptr nonnull %o)
  ret void
}

declare void @llvm.lifetime.end.p0(ptr nocapture)

define linkonce_odr dso_local void @_ZN9InterfaceC2Ev(ptr %this) unnamed_addr align 2 {
entry:
  store ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr] }, ptr @_ZTV9Interface, i64 0, i32 0, i64 2), ptr %this, align 8
  ret void
}

define linkonce_odr dso_local void @_ZN4Impl3RunEv(ptr %this) unnamed_addr align 2 {
entry:
  %ref.tmp = alloca ptr, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %ref.tmp)
  store ptr %this, ptr %ref.tmp, align 8
  call void @_Z13DoNotOptimizeIP4ImplEvRKT_(ptr nonnull dereferenceable(8) %ref.tmp)
  call void @llvm.lifetime.end.p0(ptr nonnull %ref.tmp)
  ret void
}

declare dso_local void @__cxa_pure_virtual() unnamed_addr

define linkonce_odr dso_local void @_Z13DoNotOptimizeIP4ImplEvRKT_(ptr dereferenceable(8) %var) local_unnamed_addr {
entry:
  call void asm sideeffect "", "=*m,*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(ptr) nonnull %var, ptr elementtype(ptr) nonnull %var)
  ret void
}


; Based on clang/test/CodeGenCXX/member-function-pointer-calls.cpp.
; Check that vf1 and vf2 are inlined into g1 and g2.
; CHECK: @_Z2g1v()
; CHECK-NOT: }
; CHECK: ret i32 1
; CHECK: @_Z2g2v()
; CHECK-NOT: }
; CHECK: ret i32 2
;
;struct A {
;  virtual int vf1() { return 1; }
;  virtual int vf2() { return 2; }
;};
;
;int f(A* a, int (A::*fp)()) {
;  return (a->*fp)();
;}
;int g1() {
;  A a;
;  return f(&a, &A::vf1);
;}
;int g2() {
;  A a;
;  return f(&a, &A::vf2);
;}

%struct.A = type { ptr }

@_ZTV1A = linkonce_odr unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTI1A, ptr @_ZN1A3vf1Ev, ptr @_ZN1A3vf2Ev] }, align 8
@_ZTS1A = linkonce_odr constant [3 x i8] c"1A\00", align 1
@_ZTI1A = linkonce_odr constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS1A }, align 8

define i32 @_Z1fP1AMS_FivE(ptr %a, i64 %fp.coerce0, i64 %fp.coerce1) {
entry:
  %0 = getelementptr inbounds i8, ptr %a, i64 %fp.coerce1
  %1 = and i64 %fp.coerce0, 1
  %memptr.isvirtual = icmp eq i64 %1, 0
  br i1 %memptr.isvirtual, label %memptr.nonvirtual, label %memptr.virtual

memptr.virtual:                                   ; preds = %entry
  %vtable = load ptr, ptr %0, align 8
  %2 = add i64 %fp.coerce0, -1
  %3 = getelementptr i8, ptr %vtable, i64 %2
  %memptr.virtualfn = load ptr, ptr %3, align 8
  br label %memptr.end

memptr.nonvirtual:                                ; preds = %entry
  %memptr.nonvirtualfn = inttoptr i64 %fp.coerce0 to ptr
  br label %memptr.end

memptr.end:                                       ; preds = %memptr.nonvirtual, %memptr.virtual
  %4 = phi ptr [ %memptr.virtualfn, %memptr.virtual ], [ %memptr.nonvirtualfn, %memptr.nonvirtual ]
  %call = call i32 %4(ptr %0)
  ret i32 %call
}

define i32 @_Z2g1v() {
entry:
  %a = alloca %struct.A, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %a)
  call void @_ZN1AC1Ev(ptr nonnull %a)
  %call = call i32 @_Z1fP1AMS_FivE(ptr nonnull %a, i64 1, i64 0)
  call void @llvm.lifetime.end.p0(ptr nonnull %a)
  ret i32 %call
}

define linkonce_odr void @_ZN1AC1Ev(ptr %this) align 2 {
entry:
  call void @_ZN1AC2Ev(ptr %this)
  ret void
}

define i32 @_Z2g2v() {
entry:
  %a = alloca %struct.A, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %a)
  call void @_ZN1AC1Ev(ptr nonnull %a)
  %call = call i32 @_Z1fP1AMS_FivE(ptr nonnull %a, i64 9, i64 0)
  call void @llvm.lifetime.end.p0(ptr nonnull %a)
  ret i32 %call
}

define linkonce_odr void @_ZN1AC2Ev(ptr %this) align 2 {
entry:
  store ptr getelementptr inbounds inrange(-16, 8) ({ [4 x ptr] }, ptr @_ZTV1A, i64 0, i32 0, i64 2), ptr %this, align 8
  ret void
}

define linkonce_odr i32 @_ZN1A3vf1Ev(ptr %this) align 2 {
entry:
  ret i32 1
}

define linkonce_odr i32 @_ZN1A3vf2Ev(ptr %this) align 2 {
entry:
  ret i32 2
}
