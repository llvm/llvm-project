; RUN: llvm-as < %s | llvm-dis | FileCheck %s
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
; CHECK: define void @f3() noreturn
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
; CHECK: define void @f6() nounwind
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
; CHECK: define void @f10() readnone
{
        ret void;
}

define void @f11() readonly
; CHECK: define void @f11() readonly
{
        ret void;
}

define void @f12() noinline
; CHECK: define void @f12() noinline
{
        ret void;
}

define void @f13() alwaysinline
; CHECK: define void @f13() alwaysinline
{
        ret void;
}

define void @f14() optsize
; CHECK: define void @f14() optsize
{
        ret void;
}

define void @f15() ssp
; CHECK: define void @f15() ssp
{
        ret void;
}

define void @f16() sspreq
; CHECK: define void @f16() sspreq
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
; CHECK: define void @f19() noredzone
{
        ret void;
}

define void @f20() noimplicitfloat
; CHECK: define void @f20() noimplicitfloat
{
        ret void;
}

define void @f21() naked
; CHECK: define void @f21() naked
{
        ret void;
}

define void @f22() inlinehint
; CHECK: define void @f22() inlinehint
{
        ret void;
}

define void @f23() alignstack(4)
; CHECK: define void @f23() alignstack(4)
{
        ret void;
}

define void @f24() returns_twice
; CHECK: define void @f24() returns_twice
{
        ret void;
}

define void @f25() uwtable
; CHECK: define void @f25() uwtable
{
        ret void;
}

define void @f26() nonlazybind
; CHECK: define void @f26() nonlazybind
{
        ret void;
}

define void @f27() address_safety
; CHECK: define void @f27() address_safety
{
        ret void;
}
