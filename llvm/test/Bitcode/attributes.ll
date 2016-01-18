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
; CHECK: call void @nobuiltin() #27
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
; CHECK: attributes #27 = { nobuiltin }
