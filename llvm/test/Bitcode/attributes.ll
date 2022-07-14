; RUN: llvm-as < %s | llvm-dis | FileCheck %s --check-prefixes=CHECK,CHECK-TYPED
; RUN: llvm-as -opaque-pointers < %s | llvm-dis -opaque-pointers | FileCheck %s --check-prefixes=CHECK,CHECK-OPAQUE
; RUN: verify-uselistorder < %s
; PR12696

define void @f1(i8 zeroext %0)
; CHECK: define void @f1(i8 zeroext %0)
{
        ret void;
}

define void @f2(i8 signext %0)
; CHECK: define void @f2(i8 signext %0)
{
        ret void;
}

define void @f3() noreturn
; CHECK: define void @f3() #0
{
        ret void;
}

define void @f4(i8 inreg %0)
; CHECK: define void @f4(i8 inreg %0)
{
        ret void;
}

define void @f5(i8* sret(i8) %0)
; CHECK-TYPED: define void @f5(i8* sret(i8) %0)
; CHECK-OPAQUE: define void @f5(ptr sret(i8) %0)
{
        ret void;
}

define void @f6() nounwind
; CHECK: define void @f6() #1
{
        ret void;
}

define void @f7(i8* noalias %0)
; CHECK-TYPED: define void @f7(i8* noalias %0)
; CHECK-OPAQUE: define void @f7(ptr noalias %0)
{
        ret void;
}

define void @f8(i8* byval(i8) %0)
; CHECK-TYPED: define void @f8(i8* byval(i8) %0)
; CHECK-OPAQUE: define void @f8(ptr byval(i8) %0)
{
        ret void;
}

define void @f9(i8* nest %0)
; CHECK-TYPED: define void @f9(i8* nest %0)
; CHECK-OPAQUE: define void @f9(ptr nest %0)
{
        ret void;
}

define void @f10() readnone
; CHECK: define void @f10() #2
{
        ret void;
}

define void @f11() readonly
; CHECK: define void @f11() #3
{
        ret void;
}

define void @f12() noinline
; CHECK: define void @f12() #4
{
        ret void;
}

define void @f13() alwaysinline
; CHECK: define void @f13() #5
{
        ret void;
}

define void @f14() optsize
; CHECK: define void @f14() #6
{
        ret void;
}

define void @f15() ssp
; CHECK: define void @f15() #7
{
        ret void;
}

define void @f16() sspreq
; CHECK: define void @f16() #8
{
        ret void;
}

define void @f17(i8* align 4 %0)
; CHECK-TYPED: define void @f17(i8* align 4 %0)
; CHECK-OPAQUE: define void @f17(ptr align 4 %0)
{
        ret void;
}

define void @f18(i8* nocapture %0)
; CHECK-TYPED: define void @f18(i8* nocapture %0)
; CHECK-OPAQUE: define void @f18(ptr nocapture %0)
{
        ret void;
}

define void @f19() noredzone
; CHECK: define void @f19() #9
{
        ret void;
}

define void @f20() noimplicitfloat
; CHECK: define void @f20() #10
{
        ret void;
}

define void @f21() naked
; CHECK: define void @f21() #11
{
        ret void;
}

define void @f22() inlinehint
; CHECK: define void @f22() #12
{
        ret void;
}

define void @f23() alignstack(4)
; CHECK: define void @f23() #13
{
        ret void;
}

define void @f24() returns_twice
; CHECK: define void @f24() #14
{
        ret void;
}

define void @f25() uwtable
; CHECK: define void @f25() #15
{
        ret void;
}

define void @f26() nonlazybind
; CHECK: define void @f26() #16
{
        ret void;
}

define void @f27() sanitize_address
; CHECK: define void @f27() #17
{
        ret void;
}
define void @f28() sanitize_thread
; CHECK: define void @f28() #18
{
        ret void;
}
define void @f29() sanitize_memory
; CHECK: define void @f29() #19
{
        ret void;
}

define void @f30() "cpu"="cortex-a8"
; CHECK: define void @f30() #20
{
        ret void;
}

define i8 @f31(i8 returned %A)
; CHECK: define i8 @f31(i8 returned %A)
{
        ret i8 %A;
}

define void @f32() sspstrong
; CHECK: define void @f32() #21
{
        ret void;
}

define void @f33() minsize
; CHECK: define void @f33() #22
{
        ret void;
}

declare void @nobuiltin()

define void @f34()
; CHECK: define void @f34()
{
  call void @nobuiltin() nobuiltin
; CHECK: call void @nobuiltin() #[[NOBUILTIN:[0-9]+]]
        ret void;
}

define void @f35() optnone noinline
; CHECK: define void @f35() #23
{
        ret void;
}

define void @f36(i8* inalloca(i8) %0) {
; CHECK-TYPED: define void @f36(i8* inalloca(i8) %0) {
; CHECK-OPAQUE: define void @f36(ptr inalloca(i8) %0) {
        ret void
}

define nonnull i8* @f37(i8* nonnull %a) {
; CHECK-TYPED: define nonnull i8* @f37(i8* nonnull %a) {
; CHECK-OPAQUE: define nonnull ptr @f37(ptr nonnull %a) {
        ret i8* %a
}

define void @f38() unnamed_addr jumptable {
; CHECK: define void @f38() unnamed_addr #24
    call void bitcast (void (i8*)* @f36 to void ()*)()
    unreachable
}

define dereferenceable(2) i8* @f39(i8* dereferenceable(1) %a) {
; CHECK-TYPED: define dereferenceable(2) i8* @f39(i8* dereferenceable(1) %a) {
; CHECK-OPAQUE: define dereferenceable(2) ptr @f39(ptr dereferenceable(1) %a) {
        ret i8* %a
}

define dereferenceable(18446744073709551606) i8* @f40(i8* dereferenceable(18446744073709551615) %a) {
; CHECK-TYPED: define dereferenceable(18446744073709551606) i8* @f40(i8* dereferenceable(18446744073709551615) %a) {
; CHECK-OPAQUE: define dereferenceable(18446744073709551606) ptr @f40(ptr dereferenceable(18446744073709551615) %a) {
        ret i8* %a
}

define void @f41(i8* align 32 %0, double* align 64 %1) {
; CHECK-TYPED: define void @f41(i8* align 32 %0, double* align 64 %1) {
; CHECK-OPAQUE: define void @f41(ptr align 32 %0, ptr align 64 %1) {
        ret void
}

; CHECK-TYPED: define dereferenceable_or_null(8) i8* @f42(i8* dereferenceable_or_null(8) %foo)
; CHECK-OPAQUE: define dereferenceable_or_null(8) ptr @f42(ptr dereferenceable_or_null(8) %foo)
define dereferenceable_or_null(8) i8* @f42(i8* dereferenceable_or_null(8) %foo) {
 entry:
  ret i8* %foo
}

; CHECK: define void @f43() #25
define void @f43() convergent {
  ret void
}

define void @f44() argmemonly
; CHECK: define void @f44() #26
{
        ret void;
}

; CHECK: define "string_attribute" void @f45(i32 "string_attribute" %0)
define "string_attribute" void @f45(i32 "string_attribute" %0) {
  ret void
}

; CHECK: define "string_attribute_with_value"="value" void @f46(i32 "string_attribute_with_value"="value" %0)
define "string_attribute_with_value"="value" void @f46(i32 "string_attribute_with_value"="value" %0) {
  ret void
}

; CHECK: define void @f47() #27
define void @f47() norecurse {
  ret void
}

; CHECK: define void @f48() #28
define void @f48() inaccessiblememonly {
  ret void
}

; CHECK: define void @f49() #29
define void @f49() inaccessiblemem_or_argmemonly {
  ret void
}

; CHECK-TYPED: define void @f50(i8* swiftself %0)
; CHECK-OPAQUE: define void @f50(ptr swiftself %0)
define void @f50(i8* swiftself %0)
{
  ret void;
}

; CHECK-TYPED: define i32 @f51(i8** swifterror %0)
; CHECK-OPAQUE: define i32 @f51(ptr swifterror %0)
define i32 @f51(i8** swifterror %0)
{
  ret i32 0
}

; CHECK-TYPED: define i32 @f52(i32 %0, i8** swifterror %1)
; CHECK-OPAQUE: define i32 @f52(i32 %0, ptr swifterror %1)
define i32 @f52(i32 %0, i8** swifterror %1)
{
  ret i32 0
}

%swift_error = type {i64, i8}
declare float @foo(%swift_error** swifterror %error_ptr_ref)

; CHECK: define float @f53
; CHECK: alloca swifterror
define float @f53(i8* %error_ref) {
entry:
  %error_ptr_ref = alloca swifterror %swift_error*
  store %swift_error* null, %swift_error** %error_ptr_ref
  %call = call float @foo(%swift_error** swifterror %error_ptr_ref)
  ret float 1.0
}

; CHECK-TYPED: define i8* @f54(i32 %0) #30
; CHECK-OPAQUE: define ptr @f54(i32 %0) #30
define i8* @f54(i32 %0) allocsize(0) {
  ret i8* null
}

; CHECK-TYPED: define i8* @f55(i32 %0, i32 %1) #31
; CHECK-OPAQUE: define ptr @f55(i32 %0, i32 %1) #31
define i8* @f55(i32 %0, i32 %1) allocsize(0, 1) {
  ret i8* null
}

; CHECK: define void @f56() #32
define void @f56() writeonly
{
  ret void
}

; CHECK: define void @f57() #33
define void @f57() speculatable {
  ret void
}

; CHECK: define void @f58() #34
define void @f58() sanitize_hwaddress
{
        ret void;
}

; CHECK: define void @f59() #35
define void @f59() shadowcallstack
{
  ret void
}

; CHECK: define void @f60() #36
define void @f60() willreturn
{
  ret void
}

; CHECK: define void @f61() #37
define void @f61() nofree {
  ret void
}

; CHECK: define void @f62() #38
define void @f62() nosync
{
  ret void
}

; CHECK: define void @f63() #39
define void @f63() sanitize_memtag
{
  ret void
}

; CHECK-TYPED: define void @f64(i32* preallocated(i32) %a)
; CHECK-OPAQUE: define void @f64(ptr preallocated(i32) %a)
define void @f64(i32* preallocated(i32) %a)
{
  ret void
}

; CHECK: define void @f65() #40
define void @f65() null_pointer_is_valid
{
  ret void;
}

; CHECK: define noundef i32 @f66(i32 noundef %a)
define noundef i32 @f66(i32 noundef %a)
{
  ret i32 %a
}

; CHECK-TYPED: define void @f67(i32* byref(i32) %a)
; CHECK-OPAQUE: define void @f67(ptr byref(i32) %a)
define void @f67(i32* byref(i32) %a)
{
  ret void
}

; CHECK: define void @f68() #41
define void @f68() mustprogress
{
  ret void
}

; CHECK: define void @f69() #42
define void @f69() nocallback
{
  ret void
}

; CHECK: define void @f70() #43
define void @f70() cold
{
  ret void
}

; CHECK: define void @f71() #44
define void @f71() hot
{
  ret void
}

; CHECK: define void @f72() #45
define void @f72() vscale_range(8)
{
  ret void
}

; CHECK: define void @f73() #46
define void @f73() vscale_range(1,8)
{
  ret void
}

; CHECK: define void @f74() #47
define void @f74() vscale_range(1,0)
{
  ret void
}

; CHECK-TYPED: define void @f76(i8* swiftasync %0)
; CHECK-OPAQUE: define void @f76(ptr swiftasync %0)
define void @f76(i8* swiftasync %0)
{
  ret void;
}

; CHECK: define void @f77() #48
define void @f77() nosanitize_coverage
{
        ret void;
}

; CHECK: define void @f78() #49
define void @f78() noprofile
{
        ret void;
}

declare void @llvm.some.intrinsic(i32*)
define void @f79() {
; CHECK-TYPED: call void @llvm.some.intrinsic(i32* elementtype(i32) null)
; CHECK-OPAQUE: call void @llvm.some.intrinsic(ptr elementtype(i32) null)
  call void @llvm.some.intrinsic(i32* elementtype(i32) null)
  ret void
}

; CHECK: define void @f80() #50
define void @f80() disable_sanitizer_instrumentation
{
        ret void;
}

define void @f81(i8** sret(i8*) %0)
; CHECK-TYPED: define void @f81(i8** sret(i8*) %0)
; CHECK-OPAQUE: define void @f81(ptr sret(ptr) %0)
{
        ret void;
}

define void @f82(i32* %0)
; CHECK-TYPED: define void @f82(i32* %0)
; CHECK-OPAQUE: define void @f82(ptr %0)
{
; CHECK-TYPED: call void @llvm.some.intrinsic(i32* sret(i32) %0)
; CHECK-OPAQUE: call void @llvm.some.intrinsic(ptr sret(i32) %0)
        call void @llvm.some.intrinsic(i32* sret(i32) %0)
        ret void;
}

; CHECK-TYPED: define void @f83(<4 x i8*> align 32 %0, <vscale x 1 x double*> align 64 %1)
; CHECK-OPQUE: define void @f83(<4 x ptr> align 32 %0, <vscale x 1 x ptr> align 64 %1)
define void @f83(<4 x i8*> align 32 %0, <vscale x 1 x double*> align 64 %1) {
  ret void
}

; CHECK: define void @f84() #51
define void @f84() uwtable(sync) {
        ret void;
}

; CHECK: define void @f85() #15
define void @f85() uwtable(async) {
        ret void;
}

; CHECK: define void @f86() #52
define void @f86() nosanitize_bounds
{
        ret void;
}

; CHECK: define void @f87() [[FNRETTHUNKEXTERN:#[0-9]+]]
define void @f87() fn_ret_thunk_extern { ret void }

; CHECK: attributes #0 = { noreturn }
; CHECK: attributes #1 = { nounwind }
; CHECK: attributes #2 = { readnone }
; CHECK: attributes #3 = { readonly }
; CHECK: attributes #4 = { noinline }
; CHECK: attributes #5 = { alwaysinline }
; CHECK: attributes #6 = { optsize }
; CHECK: attributes #7 = { ssp }
; CHECK: attributes #8 = { sspreq }
; CHECK: attributes #9 = { noredzone }
; CHECK: attributes #10 = { noimplicitfloat }
; CHECK: attributes #11 = { naked }
; CHECK: attributes #12 = { inlinehint }
; CHECK: attributes #13 = { alignstack=4 }
; CHECK: attributes #14 = { returns_twice }
; CHECK: attributes #15 = { uwtable }
; CHECK: attributes #16 = { nonlazybind }
; CHECK: attributes #17 = { sanitize_address }
; CHECK: attributes #18 = { sanitize_thread }
; CHECK: attributes #19 = { sanitize_memory }
; CHECK: attributes #20 = { "cpu"="cortex-a8" }
; CHECK: attributes #21 = { sspstrong }
; CHECK: attributes #22 = { minsize }
; CHECK: attributes #23 = { noinline optnone }
; CHECK: attributes #24 = { jumptable }
; CHECK: attributes #25 = { convergent }
; CHECK: attributes #26 = { argmemonly }
; CHECK: attributes #27 = { norecurse }
; CHECK: attributes #28 = { inaccessiblememonly }
; CHECK: attributes #29 = { inaccessiblemem_or_argmemonly }
; CHECK: attributes #30 = { allocsize(0) }
; CHECK: attributes #31 = { allocsize(0,1) }
; CHECK: attributes #32 = { writeonly }
; CHECK: attributes #33 = { speculatable }
; CHECK: attributes #34 = { sanitize_hwaddress }
; CHECK: attributes #35 = { shadowcallstack }
; CHECK: attributes #36 = { willreturn }
; CHECK: attributes #37 = { nofree }
; CHECK: attributes #38 = { nosync }
; CHECK: attributes #39 = { sanitize_memtag }
; CHECK: attributes #40 = { null_pointer_is_valid }
; CHECK: attributes #41 = { mustprogress }
; CHECK: attributes #42 = { nocallback }
; CHECK: attributes #43 = { cold }
; CHECK: attributes #44 = { hot }
; CHECK: attributes #45 = { vscale_range(8,8) }
; CHECK: attributes #46 = { vscale_range(1,8) }
; CHECK: attributes #47 = { vscale_range(1,0) }
; CHECK: attributes #48 = { nosanitize_coverage }
; CHECK: attributes #49 = { noprofile }
; CHECK: attributes #50 = { disable_sanitizer_instrumentation }
; CHECK: attributes #51 = { uwtable(sync) }
; CHECK: attributes #52 = { nosanitize_bounds }
; CHECK: attributes [[FNRETTHUNKEXTERN]] = { fn_ret_thunk_extern }
; CHECK: attributes #[[NOBUILTIN]] = { nobuiltin }
