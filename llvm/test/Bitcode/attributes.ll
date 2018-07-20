; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder < %s
; PR12696

define void @f1(i8 zeroext)
; CHECK: define void @f1(i8 zeroext)
{
        ret void;
}

define void @f2(i8 signext)
; CHECK: define void @f2(i8 signext)
{
        ret void;
}

define void @f3() noreturn
; CHECK: define void @f3() #0
{
        ret void;
}

define void @f4(i8 inreg)
; CHECK: define void @f4(i8 inreg)
{
        ret void;
}

define void @f5(i8* sret)
; CHECK: define void @f5(i8* sret)
{
        ret void;
}

define void @f6() nounwind
; CHECK: define void @f6() #1
{
        ret void;
}

define void @f7(i8* noalias)
; CHECK: define void @f7(i8* noalias)
{
        ret void;
}

define void @f8(i8* byval)
; CHECK: define void @f8(i8* byval)
{
        ret void;
}

define void @f9(i8* nest)
; CHECK: define void @f9(i8* nest)
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

define void @f17(i8 align 4)
; CHECK: define void @f17(i8 align 4)
{
        ret void;
}

define void @f18(i8* nocapture)
; CHECK: define void @f18(i8* nocapture)
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
; CHECK: call void @nobuiltin() #36
        ret void;
}

define void @f35() optnone noinline
; CHECK: define void @f35() #23
{
        ret void;
}

define void @f36(i8* inalloca) {
; CHECK: define void @f36(i8* inalloca) {
        ret void
}

define nonnull i8* @f37(i8* nonnull %a) {
; CHECK: define nonnull i8* @f37(i8* nonnull %a) {
        ret i8* %a
}

define void @f38() unnamed_addr jumptable {
; CHECK: define void @f38() unnamed_addr #24
    call void bitcast (void (i8*)* @f36 to void ()*)()
    unreachable
}

define dereferenceable(2) i8* @f39(i8* dereferenceable(1) %a) {
; CHECK: define dereferenceable(2) i8* @f39(i8* dereferenceable(1) %a) {
        ret i8* %a
}

define dereferenceable(18446744073709551606) i8* @f40(i8* dereferenceable(18446744073709551615) %a) {
; CHECK: define dereferenceable(18446744073709551606) i8* @f40(i8* dereferenceable(18446744073709551615) %a) {
        ret i8* %a
}

define void @f41(i8* align 32, double* align 64) {
; CHECK: define void @f41(i8* align 32, double* align 64) {
        ret void
}

; CHECK: define dereferenceable_or_null(8) i8* @f42(i8* dereferenceable_or_null(8) %foo)
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

; CHECK: define "string_attribute" void @f45(i32 "string_attribute")
define "string_attribute" void @f45(i32 "string_attribute") {
  ret void
}

; CHECK: define "string_attribute_with_value"="value" void @f46(i32 "string_attribute_with_value"="value")
define "string_attribute_with_value"="value" void @f46(i32 "string_attribute_with_value"="value") {
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

; CHECK: define void @f50(i8* swiftself)
define void @f50(i8* swiftself)
{
  ret void;
}

; CHECK: define i32 @f51(i8** swifterror)
define i32 @f51(i8** swifterror)
{
  ret i32 0
}

; CHECK: define i32 @f52(i32, i8** swifterror)
define i32 @f52(i32, i8** swifterror)
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

; CHECK: define i8* @f54(i32) #30
define i8* @f54(i32) allocsize(0) {
  ret i8* null
}

; CHECK: define i8* @f55(i32, i32) #31
define i8* @f55(i32, i32) allocsize(0, 1) {
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
; CHECK: attributes #36 = { nobuiltin }
