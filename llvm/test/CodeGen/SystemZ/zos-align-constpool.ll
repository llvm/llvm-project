; RUN: llc < %s -mtriple=s390x-ibm-zos -mcpu=z10 | FileCheck %s
; Checks that the instruction is half-word aligned, even if there is a
; string constant with odd length.

@.str.5 = constant [13 x i8] c"\82\81\A2\89\83m\A2\A3\99\89\95\87\00"

define [1 x i64] @foo(i64 %0) {
  call void @llvm.memset.p0.i64(ptr null, i8 0, i64 %0, i1 false)
  ret [1 x i64] zeroinitializer
}

declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)

; CHECK:       ENTRY .str.5
; CHECK-NEXT: .str.5 XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(EXPORT)
; CHECK-NEXT: .str.5 DS 0H
; CHECK-NEXT:  DC XL13'8281A289836DA2A39989958700'
; CHECK-NEXT:  DS 0B
; CHECK-NEXT: L#tmp0 DS 0H
; CHECK-NEXT:  xc 0(1,2),0(2)