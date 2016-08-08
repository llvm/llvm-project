; RUN: opt -S -inline -inliner-function-import-stats=basic < %s 2>&1 | FileCheck %s -check-prefix=CHECK-BASIC -check-prefix=CHECK
; RUN: opt -S -inline -inliner-function-import-stats=verbose < %s 2>&1 | FileCheck %s -check-prefix="CHECK-VERBOSE" -check-prefix=CHECK

; CHECK: ------- Dumping inliner stats for [<stdin>] -------
; CHECK-BASIC-NOT: -- List of inlined functions:
; CHECK-BASIC-NOT: -- Inlined not imported function
; CHECK-VERBOSE: -- List of inlined functions:
; CHECK-VERBOSE: Inlined not imported function [internal2]: #inlines = 6, #inlines_to_importing_module = 2
; CHECK-VERBOSE: Inlined imported function [external2]: #inlines = 4, #inlines_to_importing_module = 1
; CHECK-VERBOSE: Inlined imported function [external1]: #inlines = 3, #inlines_to_importing_module = 2
; CHECK-VERBOSE: Inlined imported function [external5]: #inlines = 1, #inlines_to_importing_module = 1
; CHECK-VERBOSE: Inlined imported function [external3]: #inlines = 1, #inlines_to_importing_module = 0

; CHECK: -- Summary:
; CHECK: All functions: 10, imported functions: 7
; CHECK: inlined functions: 5 [50% of all functions]
; CHECK: imported functions inlined anywhere: 4 [57.14% of imported functions]
; CHECK: imported functions inlined into importing module: 3 [42.86% of imported functions], remaining: 4 [57.14% of imported functions]
; CHECK: non-imported functions inlined anywhere: 1 [33.33% of non-imported functions]
; CHECK: non-imported functions inlined into importing module: 1 [33.33% of non-imported functions]

define void @internal() {
    call fastcc void @external1()
    call fastcc void @internal2()
    call coldcc void @external_big()
    ret void
}

define void @internal2() alwaysinline {
    ret void
}

define void @internal3() {
    call fastcc void @external1()
    call fastcc void @external5()
    ret void
}

define void @external1() alwaysinline !thinlto_src_module !0 {
    call fastcc void @internal2()
    call fastcc void @external2();
    ret void
}

define void @external2() alwaysinline !thinlto_src_module !1 {
    ret void
}

define void @external3() alwaysinline !thinlto_src_module !1 {
    ret void
}

define void @external4() !thinlto_src_module !1 {
    call fastcc void @external1()
    call fastcc void @external2()
    ret void
}

define void @external5() !thinlto_src_module !1 {
    ret void
}

; Assume big piece of code here. This function won't be inlined, so all the
; inlined function it will have won't affect real inlines.
define void @external_big() noinline !thinlto_src_module !1 {
; CHECK-NOT: call fastcc void @internal2()
    call fastcc void @internal2()
    call fastcc void @internal2()
    call fastcc void @internal2()
    call fastcc void @internal2()

; CHECK-NOT: call fastcc void @external2()
    call fastcc void @external2()
    call fastcc void @external2()
; CHECK-NOT: call fastcc void @external3()
    call fastcc void @external3()
    ret void
}

; It should not be imported, but it should not break anything.
define void @external_notcalled() !thinlto_src_module !0 {
    call void @external_notcalled()
    ret void
}

!0 = !{!"file.cc"}
!1 = !{!"other.cc"}
