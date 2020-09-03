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
%class.Interface = type { i32 (...)** }

@_ZTV4Impl = linkonce_odr dso_local unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTI4Impl to i8*), i8* bitcast (void (%class.Impl*)* @_ZN4Impl3RunEv to i8*)] }, align 8
@_ZTVN10__cxxabiv120__si_class_type_infoE = external dso_local global i8*
@_ZTS4Impl = linkonce_odr dso_local constant [6 x i8] c"4Impl\00", align 1
@_ZTVN10__cxxabiv117__class_type_infoE = external dso_local global i8*
@_ZTS9Interface = linkonce_odr dso_local constant [11 x i8] c"9Interface\00", align 1
@_ZTI9Interface = linkonce_odr dso_local constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([11 x i8], [11 x i8]* @_ZTS9Interface, i32 0, i32 0) }, align 8
@_ZTI4Impl = linkonce_odr dso_local constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @_ZTS4Impl, i32 0, i32 0), i8* bitcast ({ i8*, i8* }* @_ZTI9Interface to i8*) }, align 8
@_ZTV9Interface = linkonce_odr dso_local unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI9Interface to i8*), i8* bitcast (void ()* @__cxa_pure_virtual to i8*)] }, align 8

define dso_local void @_Z4Testv() local_unnamed_addr {
entry:
  %o = alloca %class.Impl, align 8
  %0 = bitcast %class.Impl* %o to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %0)
  call void @_ZN4ImplC2Ev(%class.Impl* nonnull %o)
  %1 = getelementptr inbounds %class.Impl, %class.Impl* %o, i64 0, i32 0
  call fastcc void @_ZL11IndirectRunR9Interface(%class.Interface* nonnull dereferenceable(8) %1)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %0)
  ret void
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)

define linkonce_odr dso_local void @_ZN4ImplC2Ev(%class.Impl* %this) unnamed_addr align 2 {
entry:
  %0 = getelementptr %class.Impl, %class.Impl* %this, i64 0, i32 0
  call void @_ZN9InterfaceC2Ev(%class.Interface* %0)
  %1 = getelementptr %class.Impl, %class.Impl* %this, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV4Impl, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %1, align 8
  %f = getelementptr inbounds %class.Impl, %class.Impl* %this, i64 0, i32 1
  store i32 3, i32* %f, align 8
  ret void
}

define internal fastcc void @_ZL11IndirectRunR9Interface(%class.Interface* dereferenceable(8) %o) unnamed_addr {
entry:
  %0 = bitcast %class.Interface* %o to void (%class.Interface*)***
  %vtable = load void (%class.Interface*)**, void (%class.Interface*)*** %0, align 8
  %1 = load void (%class.Interface*)*, void (%class.Interface*)** %vtable, align 8
  call void %1(%class.Interface* nonnull %o)
  ret void
}

declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)

define linkonce_odr dso_local void @_ZN9InterfaceC2Ev(%class.Interface* %this) unnamed_addr align 2 {
entry:
  %0 = getelementptr %class.Interface, %class.Interface* %this, i64 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV9Interface, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8
  ret void
}

define linkonce_odr dso_local void @_ZN4Impl3RunEv(%class.Impl* %this) unnamed_addr align 2 {
entry:
  %ref.tmp = alloca %class.Impl*, align 8
  %0 = bitcast %class.Impl** %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0)
  store %class.Impl* %this, %class.Impl** %ref.tmp, align 8
  call void @_Z13DoNotOptimizeIP4ImplEvRKT_(%class.Impl** nonnull dereferenceable(8) %ref.tmp)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0)
  ret void
}

declare dso_local void @__cxa_pure_virtual() unnamed_addr

define linkonce_odr dso_local void @_Z13DoNotOptimizeIP4ImplEvRKT_(%class.Impl** dereferenceable(8) %var) local_unnamed_addr {
entry:
  call void asm sideeffect "", "=*m,*m,~{dirflag},~{fpsr},~{flags}"(%class.Impl** nonnull %var, %class.Impl** nonnull %var)
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

%struct.A = type { i32 (...)** }

@_ZTV1A = linkonce_odr unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI1A to i8*), i8* bitcast (i32 (%struct.A*)* @_ZN1A3vf1Ev to i8*), i8* bitcast (i32 (%struct.A*)* @_ZN1A3vf2Ev to i8*)] }, align 8
@_ZTS1A = linkonce_odr constant [3 x i8] c"1A\00", align 1
@_ZTI1A = linkonce_odr constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @_ZTS1A, i32 0, i32 0) }, align 8

define i32 @_Z1fP1AMS_FivE(%struct.A* %a, i64 %fp.coerce0, i64 %fp.coerce1) {
entry:
  %0 = bitcast %struct.A* %a to i8*
  %1 = getelementptr inbounds i8, i8* %0, i64 %fp.coerce1
  %this.adjusted = bitcast i8* %1 to %struct.A*
  %2 = and i64 %fp.coerce0, 1
  %memptr.isvirtual = icmp eq i64 %2, 0
  br i1 %memptr.isvirtual, label %memptr.nonvirtual, label %memptr.virtual

memptr.virtual:                                   ; preds = %entry
  %3 = bitcast i8* %1 to i8**
  %vtable = load i8*, i8** %3, align 8
  %4 = add i64 %fp.coerce0, -1
  %5 = getelementptr i8, i8* %vtable, i64 %4
  %6 = bitcast i8* %5 to i32 (%struct.A*)**
  %memptr.virtualfn = load i32 (%struct.A*)*, i32 (%struct.A*)** %6, align 8
  br label %memptr.end

memptr.nonvirtual:                                ; preds = %entry
  %memptr.nonvirtualfn = inttoptr i64 %fp.coerce0 to i32 (%struct.A*)*
  br label %memptr.end

memptr.end:                                       ; preds = %memptr.nonvirtual, %memptr.virtual
  %7 = phi i32 (%struct.A*)* [ %memptr.virtualfn, %memptr.virtual ], [ %memptr.nonvirtualfn, %memptr.nonvirtual ]
  %call = call i32 %7(%struct.A* %this.adjusted)
  ret i32 %call
}

define i32 @_Z2g1v() {
entry:
  %a = alloca %struct.A, align 8
  %0 = bitcast %struct.A* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0)
  call void @_ZN1AC1Ev(%struct.A* nonnull %a)
  %call = call i32 @_Z1fP1AMS_FivE(%struct.A* nonnull %a, i64 1, i64 0)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0)
  ret i32 %call
}

define linkonce_odr void @_ZN1AC1Ev(%struct.A* %this) align 2 {
entry:
  call void @_ZN1AC2Ev(%struct.A* %this)
  ret void
}

define i32 @_Z2g2v() {
entry:
  %a = alloca %struct.A, align 8
  %0 = bitcast %struct.A* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0)
  call void @_ZN1AC1Ev(%struct.A* nonnull %a)
  %call = call i32 @_Z1fP1AMS_FivE(%struct.A* nonnull %a, i64 9, i64 0)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0)
  ret i32 %call
}

define linkonce_odr void @_ZN1AC2Ev(%struct.A* %this) align 2 {
entry:
  %0 = getelementptr %struct.A, %struct.A* %this, i64 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTV1A, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8
  ret void
}

define linkonce_odr i32 @_ZN1A3vf1Ev(%struct.A* %this) align 2 {
entry:
  ret i32 1
}

define linkonce_odr i32 @_ZN1A3vf2Ev(%struct.A* %this) align 2 {
entry:
  ret i32 2
}
