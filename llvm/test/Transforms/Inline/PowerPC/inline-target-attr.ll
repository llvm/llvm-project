; RUN: opt < %s -mtriple=powerpc64le-ibm-linux-gnu -S -passes=inline | FileCheck %s
; RUN: opt < %s -mtriple=powerpc64le-ibm-linux-gnu -S -passes='cgscc(inline)' | FileCheck %s

target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64le-ibm-linux-gnu"

declare void @inlined()

define void @explicit() #0 {
; CHECK-LABEL: explicit
; CHECK: entry
; CHECK-NEXT: call void @not_compatible1()
; CHECK-NEXT: call void @inlined()
entry:
    call void @not_compatible1()
    call void @compatible1()
    ret void
}

define void @not_compatible1() #1 {
entry:
    call i32 @inlined()
    ret void
}

define void @compatible1() #0 {
entry:
    call void @inlined()
    ret void 
}

define void @default() #3 {
; CHECK-LABEL: default
; CHECK: entry
; CHECK-NEXT: call void @not_compatible2()
; CHECK-NEXT: call void @inlined()
entry:
    call void @not_compatible2()
    call void @compatible2()
    ret void
}

define void @not_compatible2() #4 {
entry:
    call void @inlined()
    ret void
}

define void @compatible2() #5 {
entry:
    call void @inlined()
    ret void 
}

; explicit
attributes #0 = { "target-cpu"="pwr7" "target-features"="+allow-unaligned-fp-access" }
attributes #1 = { "target-cpu"="pwr7" "target-features"="-allow-unaligned-fp-access" }

; pwr7 by default implies +vsx
attributes #3 = { "target-cpu"="pwr7" }
attributes #4 = { "target-cpu"="pwr7" "target-features"="-vsx" }
attributes #5 = { "target-cpu"="pwr7" "target-features"="+vsx" }

