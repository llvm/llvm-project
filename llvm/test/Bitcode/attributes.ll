; RUN: llvm-as < %s | llvm-dis | FileCheck %s
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

define void @f5(ptr sret(i8) %0)
; CHECK: define void @f5(ptr sret(i8) %0)
{
        ret void;
}

define void @f6() nounwind
; CHECK: define void @f6() #1
{
        ret void;
}

define void @f7(ptr noalias %0)
; CHECK: define void @f7(ptr noalias %0)
{
        ret void;
}

define void @f8(ptr byval(i8) %0)
; CHECK: define void @f8(ptr byval(i8) %0)
{
        ret void;
}

define void @f9(ptr nest %0)
; CHECK: define void @f9(ptr nest %0)
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

define void @f17(ptr align 4 %0)
; CHECK: define void @f17(ptr align 4 %0)
{
        ret void;
}

define void @f18(ptr nocapture %0)
; CHECK: define void @f18(ptr nocapture %0)
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

define void @f36(ptr inalloca(i8) %0) {
; CHECK: define void @f36(ptr inalloca(i8) %0) {
        ret void
}

define nonnull ptr @f37(ptr nonnull %a) {
; CHECK: define nonnull ptr @f37(ptr nonnull %a) {
        ret ptr %a
}

define void @f38() unnamed_addr jumptable {
; CHECK: define void @f38() unnamed_addr #24
    call void @f36()
    unreachable
}

define dereferenceable(2) ptr @f39(ptr dereferenceable(1) %a) {
; CHECK: define dereferenceable(2) ptr @f39(ptr dereferenceable(1) %a) {
        ret ptr %a
}

define dereferenceable(18446744073709551606) ptr @f40(ptr dereferenceable(18446744073709551615) %a) {
; CHECK: define dereferenceable(18446744073709551606) ptr @f40(ptr dereferenceable(18446744073709551615) %a) {
        ret ptr %a
}

define void @f41(ptr align 32 %0, ptr align 64 %1) {
; CHECK: define void @f41(ptr align 32 %0, ptr align 64 %1) {
        ret void
}

; CHECK: define dereferenceable_or_null(8) ptr @f42(ptr dereferenceable_or_null(8) %foo)
define dereferenceable_or_null(8) ptr @f42(ptr dereferenceable_or_null(8) %foo) {
 entry:
  ret ptr %foo
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
; CHECK: define void @f50(ptr swiftself %0)
define void @f50(ptr swiftself %0)
{
  ret void;
}

; CHECK: define i32 @f51(ptr swifterror %0)
define i32 @f51(ptr swifterror %0)
{
  ret i32 0
}

; CHECK: define i32 @f52(i32 %0, ptr swifterror %1)
define i32 @f52(i32 %0, ptr swifterror %1)
{
  ret i32 0
}

%swift_error = type {i64, i8}
declare float @foo(ptr swifterror %error_ptr_ref)

; CHECK: define float @f53
; CHECK: alloca swifterror
define float @f53(ptr %error_ref) {
entry:
  %error_ptr_ref = alloca swifterror ptr
  store ptr null, ptr %error_ptr_ref
  %call = call float @foo(ptr swifterror %error_ptr_ref)
  ret float 1.0
}

; CHECK: define ptr @f54(i32 %0) #30
define ptr @f54(i32 %0) allocsize(0) {
  ret ptr null
}

; CHECK: define ptr @f55(i32 %0, i32 %1) #31
define ptr @f55(i32 %0, i32 %1) allocsize(0, 1) {
  ret ptr null
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

; CHECK: define void @f64(ptr preallocated(i32) %a)
define void @f64(ptr preallocated(i32) %a)
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

; CHECK: define void @f67(ptr byref(i32) %a)
define void @f67(ptr byref(i32) %a)
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

; CHECK: define void @f76(ptr swiftasync %0)
define void @f76(ptr swiftasync %0)
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

declare void @llvm.some.intrinsic(ptr)
define void @f79() {
; CHECK: call void @llvm.some.intrinsic(ptr elementtype(i32) null)
  call void @llvm.some.intrinsic(ptr elementtype(i32) null)
  ret void
}

; CHECK: define void @f80() #50
define void @f80() disable_sanitizer_instrumentation
{
        ret void;
}

define void @f81(ptr sret(ptr) %0)
; CHECK: define void @f81(ptr sret(ptr) %0)
{
        ret void;
}

define void @f82(ptr %0)
; CHECK: define void @f82(ptr %0)
{
; CHECK: call void @llvm.some.intrinsic(ptr sret(i32) %0)
        call void @llvm.some.intrinsic(ptr sret(i32) %0)
        ret void;
}

; CHECK: define void @f83(<4 x ptr> align 32 %0, <vscale x 1 x ptr> align 64 %1)
define void @f83(<4 x ptr> align 32 %0, <vscale x 1 x ptr> align 64 %1) {
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

; CHECK: define void @f88() [[SKIPPROFILE:#[0-9]+]]
define void @f88() skipprofile { ret void }

; CHECK: define void @f89() [[OPTDEBUG:#[0-9]+]]
define void @f89() optdebug {
        ret void;
}

; CHECK: define void @f90(ptr writable %p)
define void @f90(ptr writable %p) {
  ret void
}

; CHECK: define void @f91(ptr dead_on_unwind %p)
define void @f91(ptr dead_on_unwind %p) {
  ret void
}

; CHECK: attributes #0 = { noreturn }
; CHECK: attributes #1 = { nounwind }
; CHECK: attributes #2 = { memory(none) }
; CHECK: attributes #3 = { memory(read) }
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
; CHECK: attributes #26 = { memory(argmem: readwrite) }
; CHECK: attributes #27 = { norecurse }
; CHECK: attributes #28 = { memory(inaccessiblemem: readwrite) }
; CHECK: attributes #29 = { memory(argmem: readwrite, inaccessiblemem: readwrite) }
; CHECK: attributes #30 = { allocsize(0) }
; CHECK: attributes #31 = { allocsize(0,1) }
; CHECK: attributes #32 = { memory(write) }
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
; CHECK: attributes [[SKIPPROFILE]] = { skipprofile }
; CHECK: attributes [[OPTDEBUG]] = { optdebug }
; CHECK: attributes #[[NOBUILTIN]] = { nobuiltin }
